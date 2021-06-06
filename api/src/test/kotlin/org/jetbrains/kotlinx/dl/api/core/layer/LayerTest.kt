package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.junit.jupiter.api.Assertions
import org.tensorflow.*
import org.tensorflow.op.Ops
import java.lang.IllegalArgumentException

open class LayerTest {

    protected fun createFloatArray1D(shape: LongArray): FloatArray {
        return FloatArray(shape[0].toInt())
    }

    protected fun createFloatArray2D(shape: LongArray): Array<FloatArray> {
        return Array(shape[0].toInt()) {
            createFloatArray1D(shape.drop(1).toLongArray())
        }
    }

    protected fun createFloatArray3D(shape: LongArray): Array<Array<FloatArray>> {
        return Array(shape[0].toInt()) {
            createFloatArray2D(shape.drop(1).toLongArray())
        }
    }

    protected fun createFloatArray4D(shape: LongArray): Array<Array<Array<FloatArray>>> {
        return Array(shape[0].toInt()) {
            createFloatArray3D(shape.drop(1).toLongArray())
        }
    }

    protected fun createFloatArray5D(shape: LongArray): Array<Array<Array<Array<FloatArray>>>> {
        return Array(shape[0].toInt()) {
            createFloatArray4D(shape.drop(1).toLongArray())
        }
    }

    private fun getInputOp(tf: Ops, input: Array<*>, inputDim: Int): Operand<Float> {
        return when (inputDim) {
            1 -> tf.constant((input as Array<Float>).toFloatArray())
            2 -> tf.constant(input as Array<FloatArray>)
            3 -> tf.constant(input as Array<Array<FloatArray>>)
            4 -> tf.constant(input as Array<Array<Array<FloatArray>>>)
            5 -> tf.constant(input as Array<Array<Array<Array<FloatArray>>>>)
            else -> throw IllegalArgumentException()
        }
    }

    protected fun runLayerInEagerMode(
        layer: Layer,
        input: Array<*>,
        inputShapeArray: LongArray
    ): Tensor<Float> {
        EagerSession.create().use {
            val tf = Ops.create()
            val range = (1 until inputShapeArray.size)
            val inputShape = Shape.make(
                inputShapeArray[0],
                *(inputShapeArray.slice(range).toLongArray())
            )
            layer.build(tf, KGraph(Graph().toGraphDef()), inputShape)

            val inputOp = getInputOp(tf, input, inputShapeArray.size)
            val isTraining = tf.constant(true)
            val numberOfLosses = tf.constant(1.0f)
            val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
            return output.tensor()
        }
    }

    protected fun assertLayerComputedOutputShape(
        layer: Layer,
        inputShapeArray: LongArray,
        expectedOutputShape: LongArray
    ) {
        val range = (1 until inputShapeArray.size)
        val inputShape = Shape.make(
            inputShapeArray[0],
            *(inputShapeArray.slice(range).toLongArray())
        )
        val outputShape = layer.computeOutputShape(inputShape).toLongArray()
        Assertions.assertArrayEquals(
            expectedOutputShape,
            outputShape,
            "Computed output shape differs from expected output shape!"
        )
    }
}