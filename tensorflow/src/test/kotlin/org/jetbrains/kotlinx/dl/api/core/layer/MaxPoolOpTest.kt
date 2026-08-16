/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool1D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.tensorflow.Graph
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32

/**
 * Pins down which TensorFlow op the max-pooling layers emit.
 *
 * [org.tensorflow.op.NnOps.maxPool] emits `MaxPoolV2`, which takes the window and strides as input
 * tensors. The layers instead build a `MaxPool` node, which takes them as attributes. Apple's
 * `tensorflow-metal` plugin registers kernels for `MaxPool`/`MaxPoolGrad` but not for the V2 forms,
 * so a `MaxPoolV2` node silently falls back to the CPU and costs roughly a third of the available
 * Metal speedup on a convolutional model.
 *
 * The two ops are numerically identical, which is what most of these tests check: switching between
 * them must be invisible to every existing model.
 */
internal class MaxPoolOpTest {

    private val input = arrayOf(
        arrayOf(
            arrayOf(floatArrayOf(1.0f), floatArrayOf(8.0f), floatArrayOf(3.0f), floatArrayOf(2.0f)),
            arrayOf(floatArrayOf(4.0f), floatArrayOf(2.0f), floatArrayOf(7.0f), floatArrayOf(1.0f)),
            arrayOf(floatArrayOf(6.0f), floatArrayOf(5.0f), floatArrayOf(0.0f), floatArrayOf(9.0f)),
            arrayOf(floatArrayOf(3.0f), floatArrayOf(1.0f), floatArrayOf(2.0f), floatArrayOf(4.0f))
        )
    )

    @Test
    fun maxPool2DEmitsTheAttributeFormOfTheOp() {
        assertEquals("MaxPool", poolingOpTypeOf(defaultMaxPool2D()))
    }

    @Test
    fun maxPool1DEmitsTheAttributeFormOfTheOp() {
        assertEquals("MaxPool", poolingOpTypeOf(defaultMaxPool1D(), ::rank3Constant))
    }

    /**
     * Guards the premise of this change: if TensorFlow Java ever gains an attribute-form overload,
     * `nn.maxPool` would stop being the V2 op and the helper could be dropped.
     */
    @Test
    fun tensorFlowJavaMaxPoolStillMapsToTheV2Op() {
        assertEquals("MaxPoolV2", org.tensorflow.op.nn.MaxPool.OP_NAME)
    }

    @Test
    fun maxPool2DProducesTheSameValuesAsTheV2Op() {
        val viaLayer = runGraph { tf -> defaultMaxPool2D().build(tf, constant(tf), tf.constant(true), null) }
        val viaV2 = runGraph { tf ->
            tf.nn.maxPool(
                constant(tf),
                tf.constant(intArrayOf(1, 2, 2, 1)),
                tf.constant(intArrayOf(1, 2, 2, 1)),
                ConvPadding.VALID.paddingName
            )
        }

        assertArrayEquals(viaV2, viaLayer, "MaxPool and MaxPoolV2 must agree exactly.")
        // 2x2 VALID windows over the 4x4 input above.
        assertArrayEquals(floatArrayOf(8.0f, 7.0f, 6.0f, 9.0f), viaLayer)
    }

    @Test
    fun maxPool2DWithSamePaddingProducesTheSameValuesAsTheV2Op() {
        val poolSize = intArrayOf(1, 3, 3, 1)
        val strides = intArrayOf(1, 2, 2, 1)

        val viaLayer = runGraph { tf ->
            MaxPool2D(poolSize, strides, ConvPadding.SAME).build(tf, constant(tf), tf.constant(true), null)
        }
        val viaV2 = runGraph { tf ->
            tf.nn.maxPool(
                constant(tf), tf.constant(poolSize), tf.constant(strides), ConvPadding.SAME.paddingName
            )
        }

        assertArrayEquals(viaV2, viaLayer, "Padding handling must be identical too.")
    }

    /**
     * The attribute form has its own gradient registration; without it every model containing a
     * pooling layer would stop training.
     */
    @Test
    fun maxPool2DRemainsDifferentiable() {
        Graph().use { graph ->
            val tf = Ops.create(graph)
            val x = tf.variable(constant(tf))
            val pooled = defaultMaxPool2D().build(tf, x, tf.constant(true), null)
            val loss = tf.math.mean(pooled, tf.constant(intArrayOf(0, 1, 2, 3)))

            val gradients = tf.gradients(loss, listOf<Operand<*>>(x))

            Session(graph).use { session ->
                session.initialize()
                session.runner().fetch(gradients.dy<TFloat32>(0)).run().use { result ->
                    val gradient = result.get(0) as TFloat32
                    assertArrayEquals(longArrayOf(1, 4, 4, 1), gradient.shape().asArray())

                    // Each of the four 2x2 windows routes its gradient to exactly one input cell,
                    // so the gradient must be non-zero in exactly four places.
                    var nonZero = 0
                    for (row in 0 until 4) for (column in 0 until 4) {
                        if (gradient.getFloat(0, row.toLong(), column.toLong(), 0) != 0.0f) nonZero++
                    }
                    assertEquals(4, nonZero, "One input per pooling window should receive gradient.")
                }
            }
        }
    }

    /** Builds the layer in a graph and returns the op type of the pooling node it produced. */
    private fun poolingOpTypeOf(layer: Layer, source: (Ops) -> Operand<TFloat32> = ::constant): String {
        Graph().use { graph ->
            val tf = Ops.create(graph)
            layer.build(tf, source(tf), tf.constant(true), null)

            val poolingOps = graph.operations().asSequence()
                .map { it.type() }
                .filter { it.startsWith("MaxPool") }
                .toList()

            assertTrue(poolingOps.isNotEmpty(), "The layer produced no pooling op at all.")
            assertEquals(1, poolingOps.size, "Expected exactly one pooling op, found $poolingOps.")
            return poolingOps.single()
        }
    }

    private fun runGraph(build: (Ops) -> Operand<TFloat32>): FloatArray {
        Graph().use { graph ->
            val tf = Ops.create(graph)
            val output = build(tf)
            Session(graph).use { session ->
                session.runner().fetch(output).run().use { result ->
                    val tensor = result.get(0) as TFloat32
                    val values = FloatArray(tensor.size().toInt())
                    tensor.asRawTensor().data().asFloats().read(values)
                    return values
                }
            }
        }
    }

    /** The rank-4 input above, as a graph constant. */
    private fun constant(tf: Ops): Operand<TFloat32> = tf.constant(input)

    // Both layers declare an IntArray and an Int constructor, each with every parameter defaulted,
    // so a no-argument call is ambiguous. These name the window explicitly.
    private fun defaultMaxPool2D() = MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1), strides = intArrayOf(1, 2, 2, 1), padding = ConvPadding.VALID
    )

    private fun defaultMaxPool1D() = MaxPool1D(
        poolSize = intArrayOf(1, 2, 1), strides = intArrayOf(1, 2, 1), padding = ConvPadding.VALID
    )

    /** MaxPool1D expects rank 3: (batch, steps, channels). */
    private fun rank3Constant(tf: Ops): Operand<TFloat32> = tf.constant(
        arrayOf(
            arrayOf(
                floatArrayOf(1.0f), floatArrayOf(8.0f), floatArrayOf(3.0f), floatArrayOf(2.0f)
            )
        )
    )
}
