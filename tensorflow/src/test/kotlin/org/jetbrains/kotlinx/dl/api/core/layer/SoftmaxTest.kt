package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.layer.activation.Softmax
import org.junit.jupiter.api.Test
import org.tensorflow.Shape

internal class SoftmaxTest : ActivationLayerTest() {

    @Test
    fun defaultSoftmaxZeros() {
        val input = floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        val output = floatArrayOf(0.25f, 0.25f, 0.25f, 0.25f)

        assertActivationFunctionSameOutputShape(Softmax(), input, output)
    }

    @Test
    fun defaultSoftmax() {
        val input = floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        val output = floatArrayOf(0.0056533f, 0.04177257f, 0.11354962f, 0.83902451f)

        assertActivationFunctionSameOutputShape(Softmax(), input, output)
    }

    @Test
    fun softmax2D() {
        val input = arrayOf(
            floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f),
            floatArrayOf(-3.0f, -1.0f, 0.0f, 2.0f)
        )

        val output0 = arrayOf(
            floatArrayOf(0.0056533f, 0.04177257f, 0.11354962f, 0.83902451f),
            floatArrayOf(0.0056533f, 0.04177257f, 0.11354962f, 0.83902451f)
        )

        val output1 = arrayOf(
            floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f),
            floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f),
        )

        assertActivationFunction2D(Softmax(), input, Shape.make(2, 4), output0)
        assertActivationFunction2D(Softmax(listOf(0)), input, Shape.make(2, 4), output1)
        assertActivationFunction2D(Softmax(listOf(1)), input, Shape.make(2, 4), output0)
    }
}