package tf_api.keras.layers

import examples.PADDING_TYPE
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Initializer
import tf_api.keras.shapeFromDims

class Conv2D<T : Number>(
    private val filterShape: LongArray,
    private val strides: LongArray,
    private val activation: Activations = Activations.Relu,
    private val kernelInitializer: Initializer<T>,
    private val biasInitializer: Initializer<T>
) : Layer<T>() {
    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    private val KERNEL = "conv_kernel"
    private val KERNEL_INIT = "conv_kernelInit"
    private val BIAS = "conv_bias"
    private val BIAS_INIT = "conv_biasInit"

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        val kernelShape = shapeFromDims(*filterShape)
        val biasShape = Shape.make(filterShape.last())

        kernel = tf.variable(kernelShape, getDType())
        bias = tf.variable(biasShape, getDType())

        // Create dense kernel tensor
        kernel =
            addWeight(tf, KERNEL, tf.variable(kernelShape, dtype), KERNEL_INIT, kernelInitializer)

        // Create bias tensor
        bias = addWeight(tf, BIAS, tf.variable(biasShape, dtype), BIAS_INIT, biasInitializer)
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal = tf.nn.biasAdd(tf.nn.conv2d(input, kernel, strides.toMutableList(), PADDING_TYPE), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }
}