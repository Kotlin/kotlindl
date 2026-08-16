/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.EagerSession
import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import org.tensorflow.types.TFloat32
import org.jetbrains.kotlinx.dl.api.inference.copyTo

internal const val IRRELEVANT_INPUT_SIZE = 8

open class ActivationLayerTest {

    protected fun assertActivationFunction2D(
        layer: Layer,
        input: Array<FloatArray>,
        expectedShape: Shape,
        expected: Array<FloatArray>
    ) {
        EagerSession.create().use {
            val tf = Ops.create(it)
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)

            val output = layer.build(tf, inputOp, isTraining, numberOfLosses)
            val actualShape = shapeFromDims(*output.asTensor().shape().asArray())
            assertEquals(expectedShape, actualShape)

            assert2DArrayEquals(
                expected,
                output.to2DArray(),
                EPS
            )
        }
    }

    private fun assert2DArrayEquals(expected: Array<FloatArray>, actual: Array<FloatArray>, eps: Float) {
        expected.zip(actual).forEach { (expectedElement, actualElement) ->
            assertArrayEquals(expectedElement, actualElement, eps)
        }
    }


    protected fun assertActivationFunction(
        layer: Layer,
        input: FloatArray,
        actual: FloatArray,
        expected: FloatArray
    ) {
        val inputSize = input.size

        EagerSession.create().use {
            val tf = Ops.create(it)
            val inputOp = tf.constant(input)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.build(tf, inputOp, isTraining, numberOfLosses).asTensor()

            val expectedShape = Shape.of(
                inputSize.toLong()
            )

            val actualShape = shapeFromDims(*output.shape().asArray())
            assertEquals(expectedShape, actualShape)

            output.copyTo(actual)

            assertArrayEquals(
                expected,
                actual,
                EPS
            )
        }
    }

    protected fun assertActivationFunctionSameOutputShape(
        layer: Layer,
        input: FloatArray,
        expected: FloatArray
    ) = assertActivationFunction(layer, input, FloatArray(input.size), expected)

    protected fun assertActivationFunctionIrrelevantInputOutput(
        layer: Layer
    ) {
        val irrelevantArrayData = FloatArray(IRRELEVANT_INPUT_SIZE)
        assertActivationFunctionSameOutputShape(layer, irrelevantArrayData, irrelevantArrayData)
    }
}

private fun Operand<TFloat32>.to2DArray(): Array<FloatArray> = asOutput().asTensor().to2DArray()

private fun TFloat32.to2DArray(): Array<FloatArray> {
    require(shape().numDimensions() == 2)
    val shape = shape()
    val array: Array<FloatArray> = Array(shape.size(0).toInt()) { FloatArray(shape.size(1).toInt()) }
    copyTo(array)
    return array
}
