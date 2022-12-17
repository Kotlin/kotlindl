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
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D.Companion.EXTRA_DIM
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D.Companion.expand
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D.Companion.expandKernel
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D.Companion.withAdded
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D.Companion.withExpandedDimensions
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.util.toLongList
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.toTensorShape
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * 1D convolution transpose layer.
 *
 * This is an operation going in the opposite direction of a normal convolution:
 * it transforms a tensor shaped like an output of some convolution into tensor that has the shape of the input.
 *
 * This layer expects input data of size `(N, L, C)` where
 * ```
 * N - batch size
 * L - length of signal sequence
 * C - number of channels
 * ```
 *
 * Note: dilation values greater than 1 are not supported on cpu
 * (see https://github.com/tensorflow/tensorflow/issues/28264).
 *
 * @property [filters] dimensionality of the output space (i.e. the number of filters in the convolution)
 * @property [kernelLength] size of the convolutional kernel (one number)
 * @property [strides] strides of the convolution for each dimension of the input tensor (three numbers)
 * @property [dilations] dilations of the convolution for each dimension of the input tensor (three numbers).
 *           Currently, dilation values greater than 1 are not supported on cpu.
 * @property [activation] activation function
 * @property [kernelInitializer] initializer for the kernel
 * @property [biasInitializer] initializer for the bias
 * @property [kernelRegularizer] regularizer for the kernel
 * @property [biasRegularizer] regularizer for the bias
 * @property [activityRegularizer] regularizer function applied to the output of the layer
 * @property [padding] type of padding to use
 * @property [outputPadding] the amount of explicit padding to use (six numbers: two for each dimension).
 * @property [useBias] a flag that specifies if the bias should be used
 * @param [name] custom layer name
 *
 * @since 0.4
 */
public class Conv1DTranspose(
    public override val filters: Int = 3,
    public val kernelLength: Int = 3,
    public override val strides: IntArray = intArrayOf(1, 1, 1),
    public override val dilations: IntArray = intArrayOf(1, 1, 1),
    public override val activation: Activations = Activations.Relu,
    public override val kernelInitializer: Initializer = HeNormal(),
    public override val biasInitializer: Initializer = HeUniform(),
    public override val kernelRegularizer: Regularizer? = null,
    public override val biasRegularizer: Regularizer? = null,
    public override val activityRegularizer: Regularizer? = null,
    public override val padding: ConvPadding = ConvPadding.SAME,
    public override val outputPadding: IntArray? = null,
    public override val useBias: Boolean = true,
    name: String = ""
) : ConvTranspose(dimensions = 1, name), NoGradients {

    init {
        requireArraySize(strides, dimensions + 2, "strides")
        requireArraySize(dilations, dimensions + 2, "dilations")
        if (outputPadding != null) requireArraySize(outputPadding, 2 * (dimensions + 2), "outputPadding")
    }

    override val kernelSize: IntArray = intArrayOf(kernelLength)

    override fun convImplementation(tf: Ops, input: Operand<Float>): Operand<Float> {
        val outputShape = computeOutputShape(input.asOutput().shape()).toTensorShape()
        // implementation of a 1D convolution with a 2D convolution
        return tf.withExpandedDimensions(input) { expandedInput ->
            // expand 1D convolution parameters to use them with a 2D convolution operation
            val expandedOutputShape = expand(outputShape)
            val expandedKernel = tf.expandKernel(kernel.variable)
            val expandedStrides = expand(strides)
            val expandedDilations = expand(dilations)
            val expandedOutputPadding = outputPadding?.withAdded(EXTRA_DIM * 2, listOf(0, 0))

            // replace "-1" batch size in the output shape with the batch size of the input
            // see more in shapeWithDynamicBatchSize documentation
            val outputShapeWithBatchSize = tf.shapeWithDynamicBatchSize(expandedOutputShape, input)

            val options = buildOptions(
                expandedDilations,
                expandedOutputPadding?.withStandardPadding(
                    padding,
                    expandKernel(kernelSize),
                    expandedDilations
                )
            )

            return@withExpandedDimensions tf.nn.conv2dBackpropInput(
                outputShapeWithBatchSize,
                expandedKernel,
                expandedInput,
                expandedStrides.toLongList(),
                if (outputPadding != null) EXPLICIT else padding.paddingName,
                *options
            )
        }
    }
}