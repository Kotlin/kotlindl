package tf_api.keras.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.keras.TensorShape
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Initializer

class Dense<T : Number>(
    val outputSize: Int,
    // activation function
    private val activation: Activations = Activations.Sigmoid,
    // initializers
    private val kernelInitializer: Initializer<T>,
    private val biasInitializer: Initializer<T>
) : Layer<T>() {
    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    private val KERNEL = "kernel"
    private val KERNEL_INIT = "kernelInit"
    private val BIAS = "bias"
    private val BIAS_INIT = "biasInit"

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        val kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), outputSize.toLong())
        val biasShape = Shape.make(outputSize.toLong())


        fanIn = inputShape.size(inputShape.numDimensions() - 1).toInt()
        fanOut = outputSize

        // Create dense kernel tensor
        kernel =
            addWeight(tf, KERNEL, tf.variable(kernelShape, dtype), KERNEL_INIT, kernelInitializer)

        // Create bias tensor
        bias = addWeight(tf, BIAS, tf.variable(biasShape, dtype), BIAS_INIT, biasInitializer)


    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        return TensorShape(inputShape).replaceLast(outputSize.toLong()).toShape()
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal: Operand<T> = tf.math.add(tf.linalg.matMul(input, kernel), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }
}