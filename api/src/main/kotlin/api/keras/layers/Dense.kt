package api.keras.layers

import api.core.KGraph
import api.keras.activations.Activations
import api.keras.initializers.Initializer
import api.keras.shape.TensorShape
import api.keras.shape.numElementsInShape
import api.keras.shape.shapeToLongArray
import api.keras.util.denseBiasVarName
import api.keras.util.denseKernelVarName
import api.tensor.convertTensorToMultiDimArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

private const val KERNEL = "dense_kernel"
private const val BIAS = "dense_bias"

class Dense(
    val outputSize: Int,
    val activation: Activations = Activations.Sigmoid,
    val kernelInitializer: Initializer,
    val biasInitializer: Initializer,
    name: String = ""
) : Layer(name) {
    private lateinit var kernelShape: Shape
    private lateinit var biasShape: Shape
    // weight tensors
    private lateinit var kernel: Variable<Float>
    private lateinit var bias: Variable<Float>

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), outputSize.toLong())
        biasShape = Shape.make(outputSize.toLong())

        fanIn = inputShape.size(inputShape.numDimensions() - 1).toInt()
        fanOut = outputSize

        if (name.isNotEmpty()) {
            val kernelVariableName = denseKernelVarName(name)
            val biasVariableName = denseBiasVarName(name)

            kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
            bias = tf.withName(biasVariableName).variable(biasShape, getDType())

            kernel = addWeight(tf, kGraph, kernelVariableName, kernel, kernelInitializer)
            bias = addWeight(tf, kGraph, biasVariableName, bias, biasInitializer)
        } else {
            kernel = tf.variable(kernelShape, getDType())
            bias = tf.variable(biasShape, getDType())
            kernel = addWeight(tf, kGraph, KERNEL, kernel, kernelInitializer)
            bias = addWeight(tf, kGraph, BIAS, bias, biasInitializer)
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        return TensorShape(inputShape).replaceLast(outputSize.toLong()).toShape()
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        val signal: Operand<Float> = tf.math.add(tf.linalg.matMul(input, kernel), bias)
        return Activations.convert(activation).apply(tf, signal, name)
    }

    override fun getWeights(): List<Array<*>> {
        val result = mutableListOf<Array<*>>()

        val session = parentModel.session

        val runner = session.runner()
            .fetch(denseKernelVarName(name))
            .fetch(denseBiasVarName(name))

        val tensorList = runner.run()
        val filtersTensor = tensorList[0]
        val biasTensor = tensorList[1]

        val dstData = filtersTensor.convertTensorToMultiDimArray()
        result.add(dstData)

        val dstData2 = biasTensor.convertTensorToMultiDimArray()
        result.add(dstData2)

        return result.toList()
    }

    override fun hasActivation(): Boolean {
        return true
    }

    override fun getParams(): Int {
        return (numElementsInShape(shapeToLongArray(kernelShape)) + numElementsInShape(shapeToLongArray(biasShape))).toInt()
    }

    override fun toString(): String {
        return "Dense(outputSize=$outputSize, activation=$activation, kernelInitializer=$kernelInitializer, biasInitializer=$biasInitializer, kernelShape=$kernelShape, biasShape=$biasShape)"
    }


}