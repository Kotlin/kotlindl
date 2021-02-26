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
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.DepthwiseConv2dNative
import org.tensorflow.op.nn.DepthwiseConv2dNative.dilations
import kotlin.math.roundToInt

private const val KERNEL_VARIABLE_NAME = "depthwise_conv2d_kernel"
private const val BIAS_VARIABLE_NAME = "depthwise_conv2d_bias"

/**
 * Depthwise separable 2D convolution. (e.g. spatial convolution over images).
 *
 * Depthwise Separable convolutions consist of performing just the first step in a depthwise spatial convolution
 * (which acts on each input channel separately).
 * The `depthMultiplier` argument controls how many
 * output channels are generated per input channel in the depthwise step.
 *
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [depthMultiplier] The number of depthwise convolution output channels for each input channel.
 * The total number of depthwise convolution output channels will be equal to `numberOfChannels * depthMultiplier`.
 * @property [depthwiseInitializer] An initializer for the depthwise kernel matrix.
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [useBias] If true the layer uses a bias vector.
 * @property [name] Custom layer name.
 * @constructor Creates [DepthwiseConv2D] object.
 */
public class DepthwiseConv2D(
    public val kernelSize: LongArray = longArrayOf(3, 3),
    public val strides: LongArray = longArrayOf(1, 1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val depthMultiplier: Int = 1,
    public val depthwiseInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = ""
) : Layer(name), NoGradients {
    // weight tensors
    private lateinit var depthwiseKernel: Variable<Float>
    private var bias: Variable<Float>? = null

    // weight tensor shapes
    private lateinit var biasShape: Shape
    private lateinit var depthwiseKernelShape: Shape

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape (make warning here)
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        depthwiseKernelShape = shapeFromDims(*kernelSize, numberOfChannels, this.depthMultiplier.toLong())
        biasShape = Shape.make(numberOfChannels * this.depthMultiplier)

        // should be calculated before addWeight because it's used in calculation, need to rewrite addWEight to avoid strange behaviour
        // calculate fanIn, fanOut
        val inputDepth = numberOfChannels // amount of channels
        val outputDepth = numberOfChannels * this.depthMultiplier // amount of channels for the next layer

        fanIn = (inputDepth * kernelSize[0] * kernelSize[1]).toInt()
        fanOut = ((outputDepth * kernelSize[0] * kernelSize[1] / (strides[0].toDouble() * strides[1])).roundToInt())

        val (kernelVariableName, biasVariableName) = defineVariableNames()
        createDepthwiseConv2DVariables(tf, kernelVariableName, biasVariableName, kGraph)
    }

    private fun defineVariableNames(): Pair<String, String> {
        return if (name.isNotEmpty()) {
            Pair(depthwiseConv2dKernelVarName(name), depthwiseConv2dBiasVarName(name))
        } else {
            Pair(KERNEL_VARIABLE_NAME, BIAS_VARIABLE_NAME)
        }
    }

    private fun createDepthwiseConv2DVariables(
        tf: Ops,
        kernelVariableName: String,
        biasVariableName: String,
        kGraph: KGraph
    ) {
        depthwiseKernel = tf.withName(kernelVariableName).variable(depthwiseKernelShape, getDType())
        if (useBias) bias = tf.withName(biasVariableName).variable(biasShape, getDType())

        depthwiseKernel = addWeight(tf, kGraph, kernelVariableName, depthwiseKernel, depthwiseInitializer)
        if (useBias) bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializer)
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

        val outFilters = inputShape.size(3) * this.depthMultiplier
        // TODO: make this calculation for others dimensions conv layers https://github.com/tensorflow/tensorflow/blob/2b96f3662bd776e277f86997659e61046b56c315/tensorflow/python/keras/layers/convolutional.py#L224
        return Shape.make(inputShape.size(0), rows, cols, outFilters)
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val tfPadding = when (padding) {
            ConvPadding.SAME -> "SAME"
            ConvPadding.VALID -> "VALID"
            ConvPadding.FULL -> "FULL"
        }

        val options: DepthwiseConv2dNative.Options = dilations(dilations.toList()).dataFormat("NHWC")
        var output: Operand<Float> =
            tf.nn.depthwiseConv2dNative(
                input,
                depthwiseKernel,
                strides.toMutableList(),
                tfPadding,
                options
            )

        if (useBias) {
            output = tf.nn.biasAdd(output, bias)
        }

        return Activations.convert(activation).apply(tf, output, name)
    }

    override val weights: List<Array<*>> get() = extractDepthConv2DWeights()

    private fun extractDepthConv2DWeights(): List<Array<*>> {
        return extractWeigths(defineVariableNames().toList())
    }

    /** Returns the shape of kernel weights. */
    public val kernelShapeArray: LongArray get() = TensorShape(depthwiseKernelShape).dims()

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray get() = TensorShape(biasShape).dims()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int
        get() = (numElementsInShape(shapeToLongArray(depthwiseKernelShape)) + numElementsInShape(
            shapeToLongArray(
                biasShape
            )
        )).toInt()

    override fun toString(): String {
        return "DepthwiseConv2D(kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, dilations=${dilations.contentToString()}, activation=$activation, depthMultiplier=$depthMultiplier, depthwiseInitializer=$depthwiseInitializer, biasInitializer=$biasInitializer, padding=$padding, useBias=$useBias, depthwiseKernel=$depthwiseKernel, bias=$bias, biasShape=$biasShape, depthwiseKernelShape=$depthwiseKernelShape)"
    }
}
