/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.*
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.util.separableConv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.separableConv2dDepthwiseKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.separableConv2dPointwiseKernelVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.DepthwiseConv2dNative
import org.tensorflow.op.nn.DepthwiseConv2dNative.dilations
import kotlin.math.roundToInt

private const val DEPTHWISE_KERNEL_VARIABLE_NAME = "separable_conv2d_depthwise_kernel"
private const val POINTWISE_KERNEL_VARIABLE_NAME = "separable_conv2d_pointwise_kernel"
private const val BIAS_VARIABLE_NAME = "separable_conv2d_bias"

/**
 * 2-D convolution with separable filters.
 *
 * Performs a depthwise convolution that acts separately on channels followed by
 * a pointwise convolution that mixes channels.  Note that this is separability
 * between dimensions `[1, 2]` and `3`, not spatial separability between dimensions `1` and `2`.
 *
 * Intuitively, separable convolutions can be understood as
 * a way to factorize a convolution kernel into two smaller kernels, or as an extreme version of an Inception block.
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [depthMultiplier] The number of depthwise convolution output channels for each input channel.
 * The total number of depthwise convolution output channels will be equal to `numberOfChannels * depthMultiplier`.
 * @property [depthwiseInitializer] An initializer for the depthwise kernel matrix.
 * @property [pointwiseInitializer] An initializer for the pointwise kernel matrix.
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [depthwiseRegularizer] Regularizer function applied to the `depthwise` kernel matrix.
 * @property [pointwiseRegularizer] Regularizer function applied to the `pointwise` kernel matrix
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [useBias] If true the layer uses a bias vector.
 * @property [name] Custom layer name.
 * @constructor Creates [SeparableConv2D] object.
 *
 * @since 0.2
 */
public class SeparableConv2D(
    public val filters: Long = 32,
    public val kernelSize: LongArray = longArrayOf(3, 3),
    public val strides: LongArray = longArrayOf(1, 1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val depthMultiplier: Int = 1,
    public val depthwiseInitializer: Initializer = HeNormal(),
    public val pointwiseInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val depthwiseRegularizer: Regularizer? = null,
    public val pointwiseRegularizer: Regularizer? = null,
    public val biasRegularizer: Regularizer? = null,
    public val activityRegularizer: Regularizer? = null,
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = ""
) : Layer(name), NoGradients, ParametrizedLayer {
    override val variables: List<VariableDto>
        get() = listOfNotNull(depthwiseKernel, pointwiseKernel, bias)

    // weight tensors
    private lateinit var depthwiseKernel: VariableDto
    private lateinit var pointwiseKernel: VariableDto
    private var bias: VariableDto? = null

    init {
        requireArraySize(kernelSize, 2, "kernelSize")
        requireArraySize(strides, 4, "strides")
        requireArraySize(dilations, 4, "dilations")
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape (make warning here)
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        // should be calculated before addWeight because it's used in calculation, need to rewrite addWEight to avoid strange behaviour
        // calculate fanIn, fanOut
        val inputDepth = numberOfChannels // amount of channels
        val outputDepth = numberOfChannels * this.depthMultiplier // amount of channels for the next layer

        fanIn = (inputDepth * kernelSize[0] * kernelSize[1]).toInt()
        fanOut = ((outputDepth * kernelSize[0] * kernelSize[1] / (strides[0].toDouble() * strides[1])).roundToInt())

        createSeparableConv2DVariables(tf, numberOfChannels)
    }

    private val depthwiseKernelVariableName: String
        get() = if (name.isNotEmpty()) separableConv2dDepthwiseKernelVarName(name) else DEPTHWISE_KERNEL_VARIABLE_NAME

    private val pointwiseKernelVariableName: String
        get() = if (name.isNotEmpty()) separableConv2dPointwiseKernelVarName(name) else POINTWISE_KERNEL_VARIABLE_NAME

    private val biasVariableName: String
        get() = if (name.isNotEmpty()) separableConv2dBiasVarName(name) else BIAS_VARIABLE_NAME

    private fun createSeparableConv2DVariables(tf: Ops, numberOfChannels: Long) {
        val depthwiseKernelShape = shapeFromDims(*kernelSize, numberOfChannels, this.depthMultiplier.toLong())
        depthwiseKernel = variable(
            tf,
            depthwiseKernelVariableName,
            depthwiseKernelShape,
            fanIn,
            fanOut,
            depthwiseInitializer,
            depthwiseRegularizer
        )

        val pointwiseKernelShape = shapeFromDims(1, 1, numberOfChannels * this.depthMultiplier, filters)
        pointwiseKernel = variable(
            tf,
            pointwiseKernelVariableName,
            pointwiseKernelShape,
            fanIn,
            fanOut,
            pointwiseInitializer,
            pointwiseRegularizer
        )

        if (useBias) {
            val biasShape = Shape.make(filters)
            bias = variable(tf, biasVariableName, biasShape, fanIn, fanOut, biasInitializer, biasRegularizer)
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = convOutputLength(
            rows, kernelSize[0].toInt(), padding,
            strides[1].toInt(), dilations[1].toInt()
        )
        cols = convOutputLength(
            cols, kernelSize[1].toInt(), padding,
            strides[2].toInt(), dilations[2].toInt()
        )

        val shape = Shape.make(inputShape.size(0), rows, cols, filters)
        outputShape = TensorShape(shape)
        return shape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingName = padding.paddingName
        val depthwiseConv2DOptions: DepthwiseConv2dNative.Options = dilations(dilations.toList()).dataFormat("NHWC")

        val depthwiseOutput: Operand<Float> =
            tf.nn.depthwiseConv2dNative(
                input,
                depthwiseKernel.variable,
                strides.toMutableList(),
                paddingName,
                depthwiseConv2DOptions
            )

        val pointwiseStrides = mutableListOf(1L, 1L, 1L, 1L)

        val conv2DOptions: Conv2d.Options = Conv2d.dataFormat("NHWC")
        var output: Operand<Float> =
            tf.nn.conv2d(depthwiseOutput, pointwiseKernel.variable, pointwiseStrides, "VALID", conv2DOptions)

        bias?.let {
            output = tf.nn.biasAdd(output, it.variable)
        }

        return Activations.convert(activation).apply(tf, output, name)
    }

    override var weights: Map<String, Array<*>>
        get() = extractDepthConv2DWeights()
        set(value) = assignWeights(value)

    private fun extractDepthConv2DWeights(): Map<String, Array<*>> {
        return extractWeights(variables.map { it.name })
    }

    /** Returns the shape of kernel weights. */
    public val depthwiseShapeArray: LongArray get() = TensorShape(depthwiseKernel.shape).dims()

    /** Returns the shape of kernel weights. */
    public val pointwiseShapeArray: LongArray get() = TensorShape(pointwiseKernel.shape).dims()

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray? get() = bias?.let { TensorShape(it.shape).dims() }

    override val hasActivation: Boolean get() = true

    override fun toString(): String =
        "SeparableConv2D(kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, activation=$activation, depthwiseInitializer=$depthwiseInitializer, " +
                "biasInitializer=$biasInitializer, kernelShape=${depthwiseKernel.shape}, padding=$padding)"
}
