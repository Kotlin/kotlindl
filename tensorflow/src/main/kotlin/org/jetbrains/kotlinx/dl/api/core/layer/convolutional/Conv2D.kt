/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.TrainableLayer
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.util.toLongList
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.convBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.convKernelVarName
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv2d.dilations

/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, the `activation` is applied to the outputs as well.
 *
 * It expects input data of size `(N, H, W, C)` where
 * ```
 * N - batch size
 * H - height
 * W - width
 * C - number of channels
 * ```
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Four numbers, specifying the strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilations` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [kernelRegularizer] Regularizer function applied to the `kernel` weights matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv2D] object.
 */
public class Conv2D(
    public override val filters: Int = 32,
    public override val kernelSize: IntArray = intArrayOf(3, 3),
    public override val strides: IntArray = intArrayOf(1, 1, 1, 1),
    public override val dilations: IntArray = intArrayOf(1, 1, 1, 1),
    public override val activation: Activations = Activations.Relu,
    public override val kernelInitializer: Initializer = HeNormal(),
    public override val biasInitializer: Initializer = HeUniform(),
    public override val kernelRegularizer: Regularizer? = null,
    public override val biasRegularizer: Regularizer? = null,
    public override val activityRegularizer: Regularizer? = null,
    public override val padding: ConvPadding = ConvPadding.SAME,
    public override val useBias: Boolean = true,
    name: String = ""
) : AbstractConv(name = name), TrainableLayer {

    public override var isTrainable: Boolean = true

    public constructor(
        filters: Int = 32,
        kernelSize: Int = 3,
        strides: Int = 1,
        dilations: Int = 1,
        activation: Activations = Activations.Relu,
        kernelInitializer: Initializer = HeNormal(),
        biasInitializer: Initializer = HeUniform(),
        kernelRegularizer: Regularizer? = null,
        biasRegularizer: Regularizer? = null,
        activityRegularizer: Regularizer? = null,
        padding: ConvPadding = ConvPadding.SAME,
        useBias: Boolean = true,
        name: String = ""
    ) : this(
        filters = filters,
        kernelSize = intArrayOf(kernelSize, kernelSize),
        strides = intArrayOf(1, strides, strides, 1),
        dilations = intArrayOf(1, dilations, dilations, 1),
        activation = activation,
        kernelInitializer = kernelInitializer,
        biasInitializer = biasInitializer,
        kernelRegularizer = kernelRegularizer,
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

    override fun convImplementation(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        val options = dilations(dilations.toLongList()).dataFormat("NHWC")
        return tf.nn.conv2d(
            input,
            kernel.variable,
            strides.toLongList(),
            padding.paddingName,
            options
        )
    }

    override fun kernelVarName(name: String): String = convKernelVarName(name, dim = 2)

    override fun biasVarName(name: String): String = convBiasVarName(name, dim = 2)

    override fun toString(): String {
        return "Conv2D(name = $name, isTrainable=$isTrainable, filters=$filters, kernelSize=${kernelSize.contentToString()}, " +
                "strides=${strides.contentToString()}, dilations=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "kernelInitializer=$kernelInitializer, biasInitializer=$biasInitializer, " +
                "kernelRegularizer=$kernelRegularizer, biasRegularizer=$biasRegularizer, activityRegularizer=$activityRegularizer, " +
                "padding=$padding, useBias=$useBias, hasActivation=$hasActivation, " +
                "kernelShapeArray=${kernel.shape}, biasShapeArray=${bias?.shape})"
    }
}
