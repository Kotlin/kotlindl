/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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
import org.jetbrains.kotlinx.dl.api.core.shape.toTensorShape
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv3dBackpropInput

/**
 * 3D convolution transpose layer.
 *
 * This is an operation going in the opposite direction of a normal convolution:
 * it transforms a tensor shaped like an output of some convolution into tensor that has the shape of the input.
 *
 * This layer expects input data of size `(N, D, H, W, C)` where
 * ```
 * N - batch size
 * D - depth
 * H - height
 * W - width
 * C - number of channels
 * ```
 *
 * Note: providing explicit output padding is currently not supported.
 * Dilation values greater than 1 are not supported on cpu.
 *
 * @param [filters] dimensionality of the output space (i.e. the number of filters in the convolution)
 * @property [kernelSize] size of the convolutional kernel (three numbers)
 * @property [strides] strides of the convolution for each dimension of the input tensor (five numbers)
 * @property [dilations] dilations of the convolution for each dimension of the input tensor (five numbers).
 *           Currently, dilation values greater than 1 are not supported on cpu.
 * @param [activation] activation function
 * @param [kernelInitializer] initializer for the kernel
 * @param [biasInitializer] initializer for the bias
 * @param [kernelRegularizer] regularizer for the kernel
 * @param [biasRegularizer] regularizer for the bias
 * @param [activityRegularizer] regularizer function applied to the output of the layer
 * @param [padding] type of padding to use
 * @param [useBias] a flag that specifies if the bias should be used
 * @param [name] custom layer name
 *
 * @since 0.4
 */
public class Conv3DTranspose(
    public override val filters: Int = 3,
    public override val kernelSize: IntArray = intArrayOf(3, 3, 3),
    public override val strides: IntArray = intArrayOf(1, 1, 1, 1, 1),
    public override val dilations: IntArray = intArrayOf(1, 1, 1, 1, 1),
    public override val activation: Activations = Activations.Relu,
    public override val kernelInitializer: Initializer = HeNormal(),
    public override val biasInitializer: Initializer = HeUniform(),
    public override val kernelRegularizer: Regularizer? = null,
    public override val biasRegularizer: Regularizer? = null,
    public override val activityRegularizer: Regularizer? = null,
    public override val padding: ConvPadding = ConvPadding.SAME,
    public override val useBias: Boolean = true,
    name: String = ""
) : ConvTranspose(dimensions = 3, name), NoGradients {
    public constructor(
        filters: Int = 3,
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
        kernelSize = intArrayOf(kernelSize, kernelSize, kernelSize),
        strides = intArrayOf(1, strides, strides, strides, 1),
        dilations = intArrayOf(1, dilations, dilations, dilations, 1),
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
        requireArraySize(kernelSize, dimensions, "kernelSize")
        requireArraySize(strides, dimensions + 2, "strides")
        requireArraySize(dilations, dimensions + 2, "dilations")
    }

    override val outputPadding: IntArray? get() = null

    override fun convImplementation(tf: Ops, input: Operand<Float>): Operand<Float> {
        val outputShape = computeOutputShape(input.asOutput().shape()).toTensorShape()
        val options = Conv3dBackpropInput.dilations(dilations.toLongList()).dataFormat("NDHWC")
        return tf.nn.conv3dBackpropInput(
            tf.shapeWithDynamicBatchSize(outputShape, input),
            kernel.variable,
            input,
            strides.toLongList(),
            padding.paddingName,
            options
        )
    }
}