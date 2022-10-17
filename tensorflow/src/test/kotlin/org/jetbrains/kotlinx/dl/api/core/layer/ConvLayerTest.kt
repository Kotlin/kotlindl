/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.jetbrains.kotlinx.dl.impl.util.flattenFloats
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.op.Ops
import java.nio.FloatBuffer

open class ConvLayerTest {
    protected fun assertTensorsEquals(
        layer: Layer,
        input: Array<*>,
        expectedOutput: Array<*>
    ) {
        Graph().use { graph ->
            Session(graph).use { session ->
                val tf = Ops.create(graph)
                val inputOp = tf.constant(input.shape.toLongArray(), FloatBuffer.wrap(input.flattenFloats()))
                val isTraining = tf.constant(true)
                val numberOfLosses = tf.constant(1.0f)

                val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asOutput()
                layer.setOutputShape(output.shape())
                (layer as? ParametrizedLayer)?.initialize(session)
                session.runner().fetch(output).run().first().use { outputTensor ->
                    val outputShape = outputTensor.shape()
                    val expectedShape = expectedOutput.shape.toLongArray()
                    assertArrayEquals(expectedShape, outputShape)

                    val result = outputTensor.convertTensorToFlattenFloatArray()
                    val expected = expectedOutput.flattenFloats()
                    assertArrayEquals(expected, result, EPS)
                }
            }
        }
    }

    protected fun Array<*>.sum(): Float = fold(0.0f) { acc, arr ->
        when (arr) {
            is FloatArray -> arr.sum() + acc
            is Array<*> -> arr.sum() + acc
            else -> throw IllegalArgumentException("Cannot sum array other than Array of FloatArray")
        }
    }
}
