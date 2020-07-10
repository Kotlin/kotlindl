package api.keras.layers

import api.keras.activations.Activations
import api.keras.initializers.Initializer
import api.keras.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

class Dense<T : Number>(
    val outputSize: Int,
    // activation function
    private val activation: Activations = Activations.Sigmoid,
    // initializers
    private val kernelInitializer: Initializer<T>,
    private val biasInitializer: Initializer<T>,
    name: String = ""
) : Layer<T>() {
    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    private val KERNEL = "dense_kernel"
    private val KERNEL_INIT = "dense_kernelInit"
    private val BIAS = "dense_bias"
    private val BIAS_INIT = "dense_biasInit"

    init {
        this.name = name
    }

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        val kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), outputSize.toLong())
        val biasShape = Shape.make(outputSize.toLong())

        // TODO: refactor to logging
        println("kernelShape" + TensorShape(kernelShape).dims().contentToString())
        println("biasShape" + TensorShape(biasShape).dims().contentToString())

        fanIn = inputShape.size(inputShape.numDimensions() - 1).toInt()
        fanOut = outputSize

        if (name.isNotEmpty()) {
            val kernelVariableName = name + "_" + KERNEL
            val biasVariableName = name + "_" + BIAS
            val kernelInitName = name + "_" + KERNEL_INIT
            val biasInitName = name + "_" + BIAS_INIT

            kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
            bias = tf.withName(biasVariableName).variable(biasShape, getDType())

            kernel = addWeight(tf, kernelVariableName, kernel, kernelInitName, kernelInitializer)
            bias = addWeight(tf, biasVariableName, bias, biasInitName, biasInitializer)
        } else {
            kernel = tf.variable(kernelShape, getDType())
            bias = tf.variable(biasShape, getDType())
            kernel = addWeight(tf, KERNEL, kernel, KERNEL_INIT, kernelInitializer)
            bias = addWeight(tf, BIAS, bias, BIAS_INIT, biasInitializer)
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
}