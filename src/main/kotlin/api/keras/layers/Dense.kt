package api.keras.layers

import api.KGraph
import api.keras.activations.Activations
import api.keras.initializers.Initializer
import api.keras.shape.TensorShape
import api.keras.shape.numElementsInShape
import api.keras.shape.shapeToLongArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

private const val KERNEL = "dense_kernel"
private const val KERNEL_INIT = "dense_kernelInit"
private const val BIAS = "dense_bias"
private const val BIAS_INIT = "dense_biasInit"

class Dense<T : Number>(
    val outputSize: Int,
    // activation function
    val activation: Activations = Activations.Sigmoid,
    // initializers
    private val kernelInitializer: Initializer<T>,
    private val biasInitializer: Initializer<T>,
    name: String = ""
) : Layer<T>() {
    private lateinit var kernelShape: Shape

    private lateinit var biasShape: Shape

    // weight tensors
    private lateinit var kernel: Variable<T>

    private lateinit var bias: Variable<T>

    init {
        this.name = name
    }

    override fun defineVariables(tf: Ops, kGraph: KGraph<T>, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), outputSize.toLong())
        biasShape = Shape.make(outputSize.toLong())

        fanIn = inputShape.size(inputShape.numDimensions() - 1).toInt()
        fanOut = outputSize

        if (name.isNotEmpty()) {
            val kernelVariableName = name + "_" + KERNEL
            val biasVariableName = name + "_" + BIAS
            val kernelInitName = name + "_" + KERNEL_INIT
            val biasInitName = name + "_" + BIAS_INIT

            kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
            bias = tf.withName(biasVariableName).variable(biasShape, getDType())

            kernel = addWeight(tf, kGraph, kernelVariableName, kernel, kernelInitName, kernelInitializer)
            bias = addWeight(tf, kGraph, biasVariableName, bias, biasInitName, biasInitializer)
        } else {
            kernel = tf.variable(kernelShape, getDType())
            bias = tf.variable(biasShape, getDType())
            kernel = addWeight(tf, kGraph, KERNEL, kernel, KERNEL_INIT, kernelInitializer)
            bias = addWeight(tf, kGraph, BIAS, bias, BIAS_INIT, biasInitializer)
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        return TensorShape(inputShape).replaceLast(outputSize.toLong()).toShape()
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal: Operand<T> = tf.math.add(tf.linalg.matMul(input, kernel), bias)
        return Activations.convert<T>(activation).apply(tf, signal, name)
    }

    override fun getWeights(): List<Array<*>> {
        TODO("Not yet implemented")
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return (numElementsInShape(shapeToLongArray(kernelShape)) + numElementsInShape(shapeToLongArray(biasShape))).toInt()
    }
}