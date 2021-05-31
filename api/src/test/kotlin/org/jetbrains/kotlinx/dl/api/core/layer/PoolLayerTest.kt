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


open class PoolLayerTest {

    protected fun assertGlobalAvgPool1DEquals(
        layer: Layer,
        input:Array<Array<FloatArray>>,
        expected: Array<FloatArray>,
    ) {
        val actual = Array(expected.size) {FloatArray(expected[0].size) { 0.toFloat() } }
        assertPoolingLayer(layer,input, expected,actual,::assertGlobalAvgPool1DEquals)
    }


    private fun assertPoolingLayer(
        layer: Layer,
        input:Array<Array<FloatArray>>,
        expected: Array<FloatArray>,
        actual:Array<FloatArray>,
        assertEqual: (Array<FloatArray>, Array<FloatArray>)->Unit,
    ){
        val inputSize = input.size
        val inputShape = Shape.make(inputSize.toLong())
        EagerSession.create().use {
            val tf = Ops.create(it)
            val inputOp = tf.constant(input)
            layer.build(tf, KGraph(Graph().toGraphDef()), inputShape)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput().tensor()

            val expectedShape = Shape.make(
                expected.size.toLong(),
                expected[0].size.toLong()
            )

            val actualShape = shapeFromDims(*output.shape())
            output.copyTo(actual)
            assertEquals(expectedShape, actualShape)
            assertEqual(expected,actual)
        }

    }

    private fun assertGlobalAvgPool1DEquals(
        expected:  Array<FloatArray>,
        actual:  Array<FloatArray>
    ) {
        val expectedTensor = expected
        val actualTensor = actual
        val msg = "Expected ${expectedTensor.contentDeepToString()} " +
                "to equal ${actualTensor.contentDeepToString()}"
        for (i in expectedTensor.indices) {
            assertArrayEquals(expectedTensor[i], actualTensor[i], EPS, msg)
        }
    }
}
