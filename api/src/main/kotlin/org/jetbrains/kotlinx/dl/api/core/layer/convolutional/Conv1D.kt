/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

private const val KERNEL_VARIABLE_NAME = "conv1d_kernel"

private const val BIAS_VARIABLE_NAME = "conv1d_bias"

private const val EXTRA_DIM = 1L

/**
 * 1D convolution layer (e.g. convolution over audio data).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, the `activation` is applied to the outputs as well.
 *
 * It expects input data of size `(N, L, C)` where
 * ```
 * N - batch size
 * L - length of signal sequence
 * C - number of channels
 * ```
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Long number, specifying the width of the 1D convolution window.
 * @property [strides] Three numbers specifying stride of the pooling
 * operation for each dimension of input tensor.
 * NOTE: Specifying stride value != 1 is incompatible with specifying `dilation` value != 1.
 * @property [dilation] Three numbers specifying the dilation rate to use for
 * dilated convolution sequence dimensions of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv1D] object.
 *
 * @since 0.3
 */
public class Conv1D(
    public val filters: Long = 32,
    public val kernelSize: Long = 3,
    public val strides: LongArray = longArrayOf(1, 1, 1),
    public val dilation: LongArray = longArrayOf(1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val kernelInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = "",
) : Conv2DImpl(
    filtersInternal = filters,
    kernelSizeInternal = longArrayOf(1, kernelSize),
    stridesInternal = longArrayOf(strides[0], 1, strides[1], strides[2]),
    dilationsInternal = longArrayOf(dilation[0], 1, dilation[1], dilation[2]),
    activationInternal = activation,
    kernelInitializerInternal = kernelInitializer,
    biasInitializerInternal = biasInitializer,
    paddingInternal = padding,
    useBiasInternal = useBias,
    kernelVariableName = KERNEL_VARIABLE_NAME,
    biasVariableName = BIAS_VARIABLE_NAME,
    name = name
) {
    private val squeezeAxis = Squeeze.axis(listOf(EXTRA_DIM))

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val reshapedInput = tf.expandDims(input, tf.constant(EXTRA_DIM))
        val result = super.forward(tf, reshapedInput, isTraining, numberOfLosses)
        return tf.squeeze(result, squeezeAxis)
    }

    override fun toString(): String {
        return "Conv2D(filters=$filters, kernelSize=$kernelSize, strides=$strides, " +
                "dilation=$dilation, activation=$activation, kernelInitializer=$kernelInitializer, " +
                "biasInitializer=$biasInitializer, kernelShape=$kernelShape, biasShape=$biasShape, padding=$padding)"
    }
}
