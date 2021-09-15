package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.junit.jupiter.api.Assertions
import org.tensorflow.*
import org.tensorflow.op.Ops

enum class RunMode {
    EAGER,
    GRAPH
}

open class LayerTest {

    private fun getInputOp(tf: Ops, input: Array<*>): Operand<Float> {
        return when (input.shape.numDimensions()) {
            1 -> tf.constant(input.map { it as Float }.toFloatArray())
            2 -> tf.constant(input.cast2D<FloatArray>())
            3 -> tf.constant(input.cast3D<FloatArray>())
            4 -> tf.constant(input.cast4D<FloatArray>())
            5 -> tf.constant(input.cast5D<FloatArray>())
            else -> throw IllegalArgumentException("Inputs with more than 5 dimensions are not supported yet!")
        }
    }

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

                    if (layer is ParametrizedLayer) layer.initialize(session)

                    return session.runner().fetch(outputOp).run().first()
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
            val outputShape = shapeFromDims(*output.shape())
            val outputArray = getFloatArrayOfShape(outputShape).let {
                when (outputShape.numDimensions()) {
                    1 -> it as Array<Float>
                    2 -> it.cast2D<FloatArray>()
                    3 -> it.cast3D<FloatArray>()
                    4 -> it.cast4D<FloatArray>()
                    5 -> it.cast5D<FloatArray>()
                    else -> throw IllegalArgumentException("Arrays with more than 5 dimensions are not supported yet!")
                }
            }
            output.copyTo(outputArray)
            Assertions.assertArrayEquals(
                expectedOutput,
                outputArray
            )
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
        Assertions.assertArrayEquals(
            expectedOutputShape,
            outputShape,
            "Computed output shape differs from expected output shape!",
        )
    }
}
