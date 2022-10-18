/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.jetbrains.kotlinx.dl.impl.util.flattenFloats
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.tensorflow.*
import org.tensorflow.op.Ops
import java.nio.FloatBuffer

enum class RunMode {
    EAGER,
    GRAPH
}

open class LayerTest {
    private fun getInputOp(tf: Ops, input: Array<*>): Operand<Float> =
        tf.constant(input.shape.toLongArray(), FloatBuffer.wrap(input.flattenFloats()))

    private fun getLayerOutputOp(
        tf: Ops,
        layer: Layer,
        input: Array<*>,
    ): Output<*> {
        val inputOp = getInputOp(tf, input)
        val isTraining = tf.constant(true)
        val numberOfLosses = tf.constant(1.0f)
        val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()
        layer.setOutputShape(output.shape())
        return output
    }

    private fun runLayerInEagerMode(
        layer: Layer,
        input: Array<*>,
    ): Tensor<*> {
        EagerSession.create().use {
            val tf = Ops.create()
            val outputOp = getLayerOutputOp(tf, layer, input)
            return outputOp.tensor()
        }
    }

    private fun runLayerInGraphMode(
        layer: Layer,
        input: Array<*>,
    ): Tensor<*> {
        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                val outputOp = getLayerOutputOp(tf, layer, input)
                (layer as? ParametrizedLayer)?.initialize(session)
                return session.runner().fetch(outputOp).run().first()
            }
        }
    }

    /**
     * Checks the output of a layer given the input data is equal to the expected output.
     *
     * This takes care of building and running the layer instance ([layer]), in either of
     * Eager or Graph mode execution ([runMode]) to verify the output of layer for the given
     * input data ([input]), is equal to the expected output ([expectedOutput]).
     *
     * Note that this method could be used for a layer with any input/output dimensionality.
     */
    protected fun assertLayerOutputIsCorrect(
        layer: Layer,
        input: Array<*>,
        expectedOutput: Array<*>,
        runMode: RunMode = RunMode.EAGER,
    ) {
        val output = when (runMode) {
            RunMode.EAGER -> runLayerInEagerMode(layer, input)
            RunMode.GRAPH -> runLayerInGraphMode(layer, input)
        }
        output.use {
            val outputShape = output.shape()
            val expectedShape = expectedOutput.shape.toLongArray()
            assertArrayEquals(expectedShape, outputShape)

            val result = it.convertTensorToFlattenFloatArray()
            val expected = expectedOutput.flattenFloats()
            assertArrayEquals(expected, result)
        }
    }

    /**
     * Checks the computed output shape of layer is equal to the expected output shape.
     */
    protected fun assertLayerComputedOutputShape(layer: Layer, expectedOutputShape: LongArray) {
        assertArrayEquals(
            expectedOutputShape, layer.outputShape.dims(),
            "Computed output shape differs from expected output shape!",
        )
    }
}
