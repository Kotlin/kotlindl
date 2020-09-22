package api.keras.layers.twodim

import api.core.KGraph
import api.keras.activations.Activations
import api.keras.initializers.Initializer
import api.keras.layers.Layer
import api.keras.shape.convOutputLength
import api.keras.shape.numElementsInShape
import api.keras.shape.shapeFromDims
import api.keras.shape.shapeToLongArray
import api.keras.util.conv2dBiasVarName
import api.keras.util.conv2dKernelVarName
import api.keras.util.getDType
import api.tensor.convertTensorToMultiDimArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.Conv2d.dilations
import kotlin.math.roundToInt

/**
 * Type of padding.
 */
enum class ConvPadding {

    SAME,

    /** No padding. */
    VALID,
    FULL
}

private const val KERNEL = "conv2d_kernel"
private const val BIAS = "conv2d_bias"

/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, if `activation` is applied to the outputs as well.
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
 * @property [dilations]
 * @property [activation]
 * @property [kernelInitializer]
 * @property [biasInitializer]
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @constructor Creates [Conv2D] object.
 */
class Conv2D(
    val filters: Long,
    val kernelSize: LongArray,
    val strides: LongArray,
    val dilations: LongArray = longArrayOf(1, 1, 1, 1),
    val activation: Activations = Activations.Relu,
    val kernelInitializer: Initializer,
    val biasInitializer: Initializer,
    val padding: ConvPadding = ConvPadding.SAME,
    name: String = ""
) : Layer(name) {
    // weight tensors
    private lateinit var kernel: Variable<Float>
    private lateinit var bias: Variable<Float>

    // weight tensor shapes
    private lateinit var biasShape: Shape
    private lateinit var kernelShape: Shape

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape (make warning here)
        val lastElement = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        kernelShape = shapeFromDims(*kernelSize, lastElement, filters)
        biasShape = Shape.make(filters)

        // should be calculated before addWeight because it's used in calculation, need to rewrite addWEight to avoid strange behaviour
        // calculate fanIn, fanOut
        val inputDepth = lastElement // amount of channels
        val outputDepth = filters // amount of channels for the next layer

        fanIn = (inputDepth * kernelSize[0] * kernelSize[1]).toInt()
        fanOut = ((outputDepth * kernelSize[0] * kernelSize[1] / (strides[0].toDouble() * strides[1])).roundToInt())

        if (name.isNotEmpty()) {
            val kernelVariableName = conv2dKernelVarName(name)
            val biasVariableName = conv2dBiasVarName(name)

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
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = convOutputLength(
            rows, kernelSize[0].toInt(), padding,
            strides[1].toInt()
        )
        cols = convOutputLength(
            cols, kernelSize[1].toInt(), padding,
            strides[2].toInt()
        )

        // TODO: make this calculation for others dimensions conv layers https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/convolutional.py#L224
        return Shape.make(inputShape.size(0), rows, cols, filters)
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        val tfPadding = when (padding) {
            ConvPadding.SAME -> "SAME"
            ConvPadding.VALID -> "VALID"
            ConvPadding.FULL -> "FULL"
        }

        val options: Conv2d.Options = dilations(dilations.toList()).dataFormat("NHWC")
        val signal = tf.nn.biasAdd(tf.nn.conv2d(input, kernel, strides.toMutableList(), tfPadding, options), bias)
        return Activations.convert(activation).apply(tf, signal, name)
    }

    override fun getWeights(): List<Array<*>> {
        val result = mutableListOf<Array<*>>()

        val session = parentModel.session

        val runner = session.runner()
            .fetch(conv2dKernelVarName(name))
            .fetch(conv2dBiasVarName(name))

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
        return "Conv2D(filters=$filters, kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, dilations=${dilations.contentToString()}, activation=$activation, kernelInitializer=$kernelInitializer, biasInitializer=$biasInitializer, kernelShape=$kernelShape, padding=$padding)"
    }
}