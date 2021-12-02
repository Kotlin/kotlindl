package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.shape.flattenFloats
import org.jetbrains.kotlinx.dl.api.core.shape.shape
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.shape.toLongArray
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToFlattenFloatArray
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
        kGraph: KGraph,
    ): Output<*> {
        val inputShape = input.shape
        layer.build(tf, kGraph, inputShape)
        val inputOp = getInputOp(tf, input)
        val isTraining = tf.constant(true)
        val numberOfLosses = tf.constant(1.0f)
        val output = layer.forward(tf, inputOp, isTraining, numberOfLosses).asOutput()
        return output
    }

    private fun runLayerInEagerMode(
        layer: Layer,
        input: Array<*>,
    ): Tensor<*> {
        EagerSession.create().use {
            val tf = Ops.create()
            val kGraph = KGraph(Graph().toGraphDef())
            val outputOp = getLayerOutputOp(tf, layer, input, kGraph)
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
                KGraph(graph.toGraphDef()).use { kGraph ->
                    val outputOp = getLayerOutputOp(tf, layer, input, kGraph)
                    kGraph.initializeGraphVariables(session)
                    val outputTensor = session.runner().fetch(outputOp).run().first()
                    return outputTensor
                }
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
     *
     * Essentially, this method invokes the `computeOutputShape` of a layer instance ([layer])
     * given an input shape array ([inputShapeArray]) and verifies its output is equal to the
     * expected output shape ([expectedOutputShape]).
     */
    protected fun assertLayerComputedOutputShape(
        layer: Layer,
        inputShapeArray: LongArray,
        expectedOutputShape: LongArray,
    ) {
        val inputShape = shapeFromDims(*inputShapeArray)
        val outputShape = layer.computeOutputShape(inputShape).toLongArray()
        assertArrayEquals(
            expectedOutputShape,
            outputShape,
            "Computed output shape differs from expected output shape!",
        )
    }
}
