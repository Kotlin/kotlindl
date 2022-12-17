/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.util.toLongList
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.depthwiseConv2dKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.toLongArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.DepthwiseConv2dNative

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
 * @property [depthwiseRegularizer] Regularizer function applied to the depthwise kernel matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [useBias] If true the layer uses a bias vector.
 * @property [name] Custom layer name.
 * @constructor Creates [DepthwiseConv2D] object.
 * @since 0.2
 */
public class DepthwiseConv2D(
    public override val kernelSize: IntArray = intArrayOf(3, 3),
    public override val strides: IntArray = intArrayOf(1, 1, 1, 1),
    public override val dilations: IntArray = intArrayOf(1, 1, 1, 1),
    public override val activation: Activations = Activations.Relu,
    public val depthMultiplier: Int = 1,
    public val depthwiseInitializer: Initializer = HeNormal(),
    public override val biasInitializer: Initializer = HeUniform(),
    public val depthwiseRegularizer: Regularizer? = null,
    public override val biasRegularizer: Regularizer? = null,
    public override val activityRegularizer: Regularizer? = null,
    public override val padding: ConvPadding = ConvPadding.SAME,
    public override val useBias: Boolean = true,
    name: String = ""
) : AbstractConv(name = name), NoGradients {
    public constructor(
        kernelSize: Int = 3,
        strides: Int = 1,
        dilations: Int = 1,
        activation: Activations = Activations.Relu,
        depthMultiplier: Int = 1,
        depthwiseInitializer: Initializer = HeNormal(),
        biasInitializer: Initializer = HeUniform(),
        depthwiseRegularizer: Regularizer? = null,
        biasRegularizer: Regularizer? = null,
        activityRegularizer: Regularizer? = null,
        padding: ConvPadding = ConvPadding.SAME,
        useBias: Boolean = true,
        name: String = ""
    ) : this(
        kernelSize = intArrayOf(kernelSize, kernelSize),
        strides = intArrayOf(1, strides, strides, 1),
        dilations = intArrayOf(1, dilations, dilations, 1),
        activation = activation,
        depthMultiplier = depthMultiplier,
        depthwiseInitializer = depthwiseInitializer,
        biasInitializer = biasInitializer,
        depthwiseRegularizer = depthwiseRegularizer,
        biasRegularizer = biasRegularizer,
        activityRegularizer = activityRegularizer,
        padding = padding,
        useBias = useBias,
        name = name
    )

    init {
        requireArraySize(kernelSize, 2, "kernelSize")
        requireArraySize(strides, 4, "strides")
        requireArraySize(dilations, 4, "dilations")
    }

    // filters is not used in any place of this implementation of AbstractConv because
    // all its usages are overridden with custom functions that use the depthMultiplier and the
    // shape of the input data representing number of channels in it
    override val filters: Int get() = -1
    override val kernelInitializer: Initializer get() = depthwiseInitializer
    override val kernelRegularizer: Regularizer? get() = depthwiseRegularizer

    protected override fun computeKernelShape(numberOfChannels: Long): Shape =
        shapeFromDims(*kernelSize.toLongArray(), numberOfChannels, depthMultiplier.toLong())

    protected override fun computeBiasShape(numberOfChannels: Long): Shape =
        Shape.make(numberOfChannels * depthMultiplier)

    override fun getOutputDepth(numberOfChannels: Long): Long = numberOfChannels * depthMultiplier

    override fun kernelVarName(name: String): String = depthwiseConv2dKernelVarName(name)

    override fun biasVarName(name: String): String = depthwiseConv2dBiasVarName(name)

    override fun convImplementation(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        val options = DepthwiseConv2dNative.dilations(dilations.toLongList()).dataFormat("NHWC")
        return tf.nn.depthwiseConv2dNative(input, kernel.variable, strides.toLongList(), padding.paddingName, options)
    }

    override fun toString(): String {
        return "DepthwiseConv2D(name = $name, " +
                "kernelSize=${kernelSize.contentToString()}, " +
                "strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "depthMultiplier=$depthMultiplier, " +
                "depthwiseInitializer=$depthwiseInitializer, " +
                "biasInitializer=$biasInitializer, " +
                "depthwiseRegularizer=$depthwiseRegularizer, " +
                "biasRegularizer=$biasRegularizer, " +
                "activityRegularizer=$activityRegularizer, " +
                "padding=$padding, " +
                "useBias=$useBias, " +
                "hasActivation=$hasActivation, " +
                "depthwiseKernelShapeArray=${kernel.shape}, " +
                "biasShapeArray=${bias?.shape})"
    }
}
