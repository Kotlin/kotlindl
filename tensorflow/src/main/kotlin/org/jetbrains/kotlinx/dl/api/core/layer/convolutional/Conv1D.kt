/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.util.convBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.convKernelVarName
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze
import org.tensorflow.op.nn.Conv2d

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
 * @property [kernelLength] Long number, specifying the width of the 1D convolution window.
 * @property [strides] Three numbers specifying the strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying stride value != 1 is incompatible with specifying `dilation` value != 1.
 * @property [dilations] Three numbers specifying the dilation rate to use for
 * dilated convolution sequence dimensions of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [kernelRegularizer] Regularizer function applied to the `kernel` weights matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv1D] object.
 *
 * @since 0.3
 */
public class Conv1D(
    public override val filters: Int = 32,
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
    public override val useBias: Boolean = true,
    name: String = "",
) : AbstractConv(name = name), TrainableLayer {
    public constructor(
        filters: Int = 32,
        kernelLength: Int = 3,
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
        kernelLength = kernelLength,
        strides = intArrayOf(1, strides, 1),
        dilations = intArrayOf(1, dilations, 1),
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
        requireArraySize(strides, 3, "strides")
        requireArraySize(dilations, 3, "dilations")
    }

    public override var isTrainable: Boolean = true

    override val kernelSize: IntArray = intArrayOf(kernelLength)

    override fun kernelVarName(name: String): String = convKernelVarName(name, dim = 1)

    override fun biasVarName(name: String): String = convBiasVarName(name, dim = 1)

    override fun convImplementation(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        return tf.withExpandedDimensions(input) { expandedInput ->
            val options = Conv2d.dilations(expand(dilations).toLongList()).dataFormat("NHWC")
            return@withExpandedDimensions tf.nn.conv2d(
                expandedInput, tf.expandKernel(kernel.variable), expand(strides).toLongList(),
                padding.paddingName, options
            )
        }
    }

    override fun toString(): String {
        return "Conv1D(name = $name, isTrainable=$isTrainable, filters=$filters, kernelSize=${kernelSize.contentToString()}, " +
                "strides=${strides.contentToString()}, dilations=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "kernelInitializer=$kernelInitializer, biasInitializer=$biasInitializer, " +
                "kernelRegularizer=$kernelRegularizer, biasRegularizer=$biasRegularizer, activityRegularizer=$activityRegularizer, " +
                "padding=$padding, useBias=$useBias, hasActivation=$hasActivation, " +
                "kernelShapeArray=${kernel.shape}, biasShapeArray=${bias?.shape})"
    }

    internal companion object {
        internal const val EXTRA_DIM = 1

        /** Axis of height for which the extra dimension is added (unsqueezed) before actual
         * convolution operation and the output from actual implementation are squeezed. */
        private val squeezeAxis = Squeeze.axis(listOf(EXTRA_DIM.toLong()))

        internal fun expandKernel(kernel: IntArray): IntArray {
            return kernel.withAdded(EXTRA_DIM - 1, 1)
        }

        internal fun Ops.expandKernel(kernel: Operand<Float>): Operand<Float> {
            return expandDims(kernel, constant(EXTRA_DIM - 1))
        }

        internal fun expand(tensorShape: TensorShape): TensorShape {
            return TensorShape(tensorShape.dims().withAdded(EXTRA_DIM, 1))
        }

        internal fun expand(array: IntArray): IntArray {
            return array.withAdded(EXTRA_DIM, 1)
        }

        /**
         * Adds an extra dimension to the input, performs the provided operation
         * and squeezes the result by removing the dimension added previously.
         * This allows to perform 2D operations on 1D inputs.
         */
        internal fun Ops.withExpandedDimensions(
            input: Operand<Float>,
            operation: (Operand<Float>) -> Operand<Float>
        ): Operand<Float> {
            val expandedInput = expandDims(input, constant(EXTRA_DIM))
            val expandedOutput = operation(expandedInput)
            return squeeze(expandedOutput, squeezeAxis)
        }

        internal fun LongArray.withAdded(position: Int, element: Long): LongArray {
            return toMutableList().apply { add(position, element) }.toLongArray()
        }

        internal fun IntArray.withAdded(position: Int, element: Int): IntArray {
            return toMutableList().apply { add(position, element) }.toIntArray()
        }

        internal fun IntArray.withAdded(position: Int, elements: List<Int>): IntArray {
            return toMutableList().apply { addAll(position, elements) }.toIntArray()
        }
    }
}
