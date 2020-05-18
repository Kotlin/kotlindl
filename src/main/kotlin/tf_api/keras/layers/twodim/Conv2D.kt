package tf_api.keras.layers.twodim

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Initializer
import tf_api.keras.layers.Layer
import tf_api.keras.shape.TensorShape
import tf_api.keras.shape.shapeFromDims
import kotlin.math.roundToInt


enum class ConvPadding {
    SAME,
    VALID
}

class Conv2D<T : Number>(
    private val filters: Long,
    private val kernelSize: LongArray,
    private val strides: LongArray,
    private val activation: Activations = Activations.Relu,
    private val kernelInitializer: Initializer<T>,
    private val biasInitializer: Initializer<T>,
    private val padding: ConvPadding
) : Layer<T>() {

    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    private val KERNEL = "conv_kernel"
    private val KERNEL_INIT = "conv_kernelInit"
    private val BIAS = "conv_bias"
    private val BIAS_INIT = "conv_biasInit"

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape (make warning here)
        val lastElement = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        val kernelShape = shapeFromDims(*kernelSize, lastElement, filters)
        val biasShape = Shape.make(filters)

        println("kernelShape" + TensorShape(kernelShape).dims().contentToString())
        println("biasShape" + TensorShape(biasShape).dims().contentToString())

        kernel = tf.variable(kernelShape, getDType())
        bias = tf.variable(biasShape, getDType())

        // calculate fanIn, fanOut
        val inputDepth = lastElement // amount of channels
        val outputDepth = filters // amount of channels for the next layer

        fanIn = (inputDepth * kernelSize[0] * kernelSize[1]).toInt()
        fanOut = ((outputDepth * kernelSize[0] * kernelSize[1] / (strides[0].toDouble() * strides[1])).roundToInt())

        // Create dense kernel tensor
        kernel =
            addWeight(tf, KERNEL, tf.variable(kernelShape, dtype), KERNEL_INIT, kernelInitializer)

        // Create bias tensor
        bias = addWeight(tf, BIAS, tf.variable(biasShape, dtype), BIAS_INIT, biasInitializer)


    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        //TODO: outputShape calculation depending on padding type https://github.com/keras-team/keras/blob/master/keras/utils/conv_utils.py

        return Shape.make(inputShape.size(0), inputShape.size(1), inputShape.size(2), filters)
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val tfPadding = when (padding) {
            ConvPadding.SAME -> "SAME"
            ConvPadding.VALID -> "VALID"
        }

        val signal = tf.nn.biasAdd(tf.nn.conv2d(input, kernel, strides.toMutableList(), tfPadding), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }
}