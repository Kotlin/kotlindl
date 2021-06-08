package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.EPS
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.junit.jupiter.api.Assertions.assertArrayEquals
import org.junit.jupiter.api.Assertions.assertEquals
import org.tensorflow.EagerSession
import org.tensorflow.Graph
import org.tensorflow.*
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant

open class PoolLayerTest {
    protected fun assertGlobalAvgPool1DEquals(
        layer: Layer,
        input: Array<Array<FloatArray>>,
        expected: Array<FloatArray>,
    ) {
        val actual = Array(expected.size) { FloatArray(expected[0].size) { 0f } }
        assertPoolingLayer(layer, input, expected, actual, ::assertGlobalAvgPoolEquals) { tf, tensor ->
            tf.constant(
                tensor.cast3DArray()
            )
        }
    }

    protected fun assertGlobalAvgPool2DEquals(
        layer: Layer,
        input: Array<Array<Array<FloatArray>>>,
        expected: Array<FloatArray>,
    ) {
        val actual = Array(expected.size) { FloatArray(expected[0].size) { 0f } }
        assertPoolingLayer(layer, input, expected, actual, ::assertGlobalAvgPoolEquals) { tf, tensor ->
            tf.constant(
                tensor.cast4DArray()
            )
        }
    }

    protected fun assertGlobalAvgPool3DEquals(
        layer: Layer,
        input: Array<Array<Array<Array<FloatArray>>>>,
        expected: Array<FloatArray>,
    ) {
        val actual = Array(expected.size) { FloatArray(expected[0].size) { 0f } }
        assertPoolingLayer(layer, input, expected, actual, ::assertGlobalAvgPoolEquals) { tf, tensor ->
            tf.constant(
                tensor.cast5DArray()
            )
        }
    }

    private fun assertPoolingLayer(
        layer: Layer,
        input: Array<*>,
        expected: Array<*>,
        actual: Array<*>,
        assertEqual: (Array<*>, Array<*>) -> Unit,
        constProducer: (Ops, Array<*>) -> Constant<Float>
    ) {
        val inputSize = input.size
        val inputShape = Shape.make(inputSize.toLong())
        EagerSession.create().use {
            val tf = Ops.create(it)
            val inputOp = constProducer(tf, input)
            layer.build(tf, KGraph(Graph().toGraphDef()), inputShape)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = getShapeOfArray(expected)

            val actualShape = shapeFromDims(*output.shape())
            output.copyTo(actual)
            assertEquals(expectedShape, actualShape)
            assertEqual(expected, actual)
        }
    }

    private fun assertGlobalAvgPoolEquals(
        expected: Array<*>,
        actual: Array<*>
    ) {
        val expectedTensor = expected.cast2DArray()
        val actualTensor = actual.cast2DArray()
        val msg = "Expected ${expectedTensor.contentDeepToString()} " +
                "to equal ${actualTensor.contentDeepToString()}"
        for (i in expectedTensor.indices) {
            assertArrayEquals(expectedTensor[i], actualTensor[i], EPS, msg)
        }
    }
}
