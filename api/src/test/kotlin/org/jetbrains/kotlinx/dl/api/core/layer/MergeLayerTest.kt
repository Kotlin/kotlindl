package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.merge.*
import org.jetbrains.kotlinx.dl.api.core.shape.flattenFloats
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import org.tensorflow.*
import org.tensorflow.op.Ops
import java.nio.FloatBuffer

internal class MergeLayerTest {

    private fun getInputOp(tf:Ops, input:Array<*>): Operand<Float> =
        tf.constant(input.shape.toLongArray(), FloatBuffer.wrap(input.flattenFloats()))

    private fun getLayerOutputOp(
        tf: Ops,
        layer: AbstractMerge,
        input: List<Array<*>>,
        kGraph: KGraph,
    ): Output<*> {
        val inputShape = input.first().shape
        layer.build(tf, kGraph, inputShape)
        val inputOp = mutableListOf<Operand<Float>>()
        input.forEach { inputOp.add(getInputOp(tf,it)) }
        val isTraining = tf.constant(true)
        val numberOfLosses = tf.constant(1.0f)
        val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
        return output
    }

    private fun runLayerInEagerMode(
        layer: AbstractMerge,
        input: List<Array<*>>,
    ): Tensor<*> {
        EagerSession.create().use {
            val tf = Ops.create()
            val kGraph = KGraph(Graph().toGraphDef())
            val outputOp = getLayerOutputOp(tf, layer, input, kGraph)
            return outputOp.tensor()
        }
    }

    protected fun assertLayerOutputIsCorrect(
        layer:AbstractMerge,
        input: List<Array<*>>,
        expectedOutput: Array<*>,
    ) {
        val output = runLayerInEagerMode(layer,input)
        output.use {
            val outputShape = output.shape()
            val expectedShape = expectedOutput.shape.toLongArray()
            Assertions.assertArrayEquals(expectedShape, outputShape)
            val result = it.convertTensorToFlattenFloatArray()
            result.forEach { i -> println(i) }
            val expected = expectedOutput.flattenFloats()
            Assertions.assertArrayEquals(expected, result)
        }
    }

    @Test
    fun add() {
        val x1 = Array(1){ FloatArray(10) { it.toFloat() } }
        val x2 = Array(1){ FloatArray(10) { it.toFloat() } }
        val input = listOf(x1, x2)
        val expected = Array(1) { FloatArray(10) { 2 * it.toFloat() } }
        assertLayerOutputIsCorrect(Add(), input, expected)
    }

    @Test
    fun subtract() {
        val x1 = Array(1){ FloatArray(10) { it.toFloat() } }
        val x2 = Array(1){ FloatArray(10) { it.toFloat() } }
        val input = listOf(x1, x2)
        val expected = Array(1) { FloatArray(10) { 0f } }
        assertLayerOutputIsCorrect(Subtract(), input, expected)
    }

    @Test
    fun average() {
        val x1 =  Array(2) { FloatArray(2) { 0f } }
        val x2 = Array(2){ FloatArray(2) { 1f } }
        val input = listOf(x1, x2)
        val expected = Array(2) { FloatArray(2) { 0.5f } }
        assertLayerOutputIsCorrect(Average(), input, expected)
    }

    @Test
    fun concat(){}

    @Test
    fun maximum(){
        val x1 =  Array(5) { FloatArray(1) { it.toFloat() } }
        val x2 = Array(5){ FloatArray(1) { it.toFloat()+5 } }
        val input = listOf(x1, x2)
        val expected = Array(5) { FloatArray(1) { it.toFloat()+5 } }
        assertLayerOutputIsCorrect(Maximum(), input, expected)
    }

    @Test
    fun minimum(){
        val x1 =  Array(5) { FloatArray(1) { it.toFloat() } }
        val x2 = Array(5){ FloatArray(1) { it.toFloat()+5 } }
        val input = listOf(x1, x2)
        val expected = Array(5) { FloatArray(1) { it.toFloat() } }
        assertLayerOutputIsCorrect(Minimum(), input, expected)
    }

    @Test
    fun multiply(){
        val x1 =  Array(2) { FloatArray(2) { 0f } }
        val x2 = Array(2){ FloatArray(2) { 1f } }
        val input = listOf(x1, x2)
        val expected = Array(2) { FloatArray(2) { 0f } }
        assertLayerOutputIsCorrect(Multiply(), input, expected)
    }

    @Test
    fun dot(){
        val x = arrayOf(
            arrayOf(
                floatArrayOf(0f, 1f),
                floatArrayOf(2f, 3f),
                floatArrayOf(4f, 5f),
                floatArrayOf(6f, 7f),
                floatArrayOf(8f, 9f)
            )
        )
        val y = arrayOf(arrayOf(floatArrayOf(10f, 11f, 12f, 14f, 15f), floatArrayOf(15f, 16f, 17f, 18f, 19f)))
        val input = listOf(x, y)
        val expected = arrayOf(arrayOf(floatArrayOf(260f, 360f), floatArrayOf(320f, 445f)))
        assertLayerOutputIsCorrect(Dot(axis = intArrayOf(1, 2)), input, expected)
    }
}
