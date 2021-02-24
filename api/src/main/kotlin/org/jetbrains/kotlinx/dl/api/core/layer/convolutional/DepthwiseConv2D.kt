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
import org.jetbrains.kotlinx.dl.api.core.layer.ForwardLayer
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.DepthwiseConv2dNative
import org.tensorflow.op.nn.DepthwiseConv2dNative.dilations
import kotlin.math.roundToInt

private const val KERNEL = "depthwise_conv2d_kernel"
private const val BIAS = "depthwise_conv2d_bias"

/**
 * Depthwise 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, if `activation` is applied to the outputs as well.
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilation_rate` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
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
) : Layer(name), ForwardLayer {
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

        if (name.isNotEmpty()) {
            val kernelVariableName = depthwiseConv2dKernelVarName(name)
            val biasVariableName = depthwiseConv2dBiasVarName(name)

            depthwiseKernel = tf.withName(kernelVariableName).variable(depthwiseKernelShape, getDType())
            if (useBias) bias = tf.withName(biasVariableName).variable(biasShape, getDType())

            depthwiseKernel = addWeight(tf, kGraph, kernelVariableName, depthwiseKernel, depthwiseInitializer)
            if (useBias) bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializer)
        } else {
            depthwiseKernel = tf.variable(depthwiseKernelShape, getDType())
            if (useBias) bias = tf.variable(biasShape, getDType())
            depthwiseKernel = addWeight(tf, kGraph, KERNEL, depthwiseKernel, depthwiseInitializer)
            if (useBias) bias = addWeight(tf, kGraph, BIAS, bias!!, biasInitializer)
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
        val session = parentModel.session

        val runner = session.runner()
            .fetch(depthwiseConv2dKernelVarName(name))
            .fetch(depthwiseConv2dBiasVarName(name))

        val tensorList = runner.run()
        val filtersTensor = tensorList[0]
        val biasTensor = tensorList[1]

        return listOf(
            filtersTensor.convertTensorToMultiDimArray(),
            biasTensor.convertTensorToMultiDimArray(),
        )
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
        return "DepthwiseConv2D(kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, dilations=${dilations.contentToString()}, activation=$activation, depthwiseInitializer=$depthwiseInitializer, biasInitializer=$biasInitializer, kernelShape=$depthwiseKernelShape, padding=$padding)"
    }
}
