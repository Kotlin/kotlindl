/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.util.toLongList
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.convTransposeOutputLength
import org.jetbrains.kotlinx.dl.api.core.shape.convTransposePadding
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.util.convTransposeBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.convTransposeKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.toLongArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.Conv2dBackpropInput

/**
 * A base class for defining transposed convolution layers (sometimes called deconvolution) of different dimensions.
 *
 * This is an operation going in the opposite direction of a normal convolution:
 * it transforms a tensor shaped like an output of some convolution into tensor that has the shape of the input.
 *
 * @property [dimensions] dimensionality of this convolution operation
 * @property [outputPadding] the amount of padding to use for each dimension of the input tensor
 * @param [name] custom layer name
 */
public abstract class ConvTranspose(
    public val dimensions: Int,
    name: String = ""
) : AbstractConv(name = name) {

    protected abstract val outputPadding: IntArray?

    /**
     * Computes the output shape of the layer given the input shape.
     */
    protected fun computeOutputShape(inputShape: Shape): Shape {
        val shapes = (kernelSize.indices).map {
            convTransposeOutputLength(
                inputShape.size(it + 1),
                kernelSize[it],
                padding,
                outputPadding?.get(2 * (it + 1)),
                outputPadding?.get(2 * (it + 1) + 1),
                strides[it + 1],
                dilations[it + 1]
            )
        }
        return Shape.make(inputShape.size(0), *(shapes + filters.toLong()).toLongArray())
    }

    override fun kernelVarName(name: String): String = convTransposeKernelVarName(name, dimensions)
    override fun biasVarName(name: String): String = convTransposeBiasVarName(name, dimensions)

    protected override fun computeKernelShape(numberOfChannels: Long): Shape {
        return shapeFromDims(*kernelSize.toLongArray(), filters.toLong(), numberOfChannels)
    }

    override fun toString(): String {
        return "Conv${dimensions}DTranspose(" +
                "filters=$filters, " +
                "kernelSize=${kernelSize.contentToString()}, " +
                "kernelShape=${kernel.shape}, " +
                "biasShape=${bias?.shape}, " +
                "strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, " +
                "activation=$activation, " +
                "kernelInitializer=$kernelInitializer, " +
                "biasInitializer=$biasInitializer, " +
                "biasRegularizer=$biasRegularizer, " +
                "kernelRegularizer=$kernelRegularizer, " +
                "activityRegularizer=$activityRegularizer, " +
                "padding=$padding, " +
                "outputPadding=${outputPadding?.contentToString()} " +
                ")"
    }

    internal companion object {
        /**
         * Padding constant to indicate that specific padding values are going to be provided.
         */
        internal const val EXPLICIT = "EXPLICIT"


        /**
         * Combines explicitly provided padding value with the standard padding from the provided padding method.
         * This is needed since [org.tensorflow.op.NnOps.conv2dBackpropInput] function does not support specifying
         * both padding method and explicit output padding at the same time.
         */
        internal fun IntArray.withStandardPadding(
            padding: ConvPadding,
            kernelSize: IntArray,
            dilations: IntArray
        ): IntArray {
            val withStandardPadding = kernelSize.indices.flatMap { dim ->
                convTransposePadding(
                    padding,
                    this[2 * dim],
                    this[2 * dim + 1],
                    kernelSize[dim],
                    dilations[dim + 1]
                )
            }
            return intArrayOf(0, 0, *(withStandardPadding.toIntArray()), 0, 0)
        }

        /**
         * Builds options to pass dilations and output padding to the [org.tensorflow.op.NnOps.conv2dBackpropInput].
         */
        internal fun buildOptions(dilations: IntArray, outputPadding: IntArray?): Array<Conv2dBackpropInput.Options> {
            val options = mutableListOf(Conv2dBackpropInput.dilations(dilations.toLongList()))
            if (outputPadding != null) {
                options.add(Conv2dBackpropInput.explicitPaddings(outputPadding.toLongList()))
            }
            return options.map { it.dataFormat("NHWC") }.toTypedArray()
        }

        /**
         * Creates an integer vector with the contents of [tensorShape], except for the first dimension (batch size):
         * batch size from the [input] is used instead.
         * This is needed as [org.tensorflow.op.NnOps.conv2dBackpropInput] and [org.tensorflow.op.NnOps.conv3dBackpropInput]
         * need to have an exact shape provided, including batch size.
         * Typically, when the layer is built, batch size is not known and [tensorShape] contains a "-1" instead.
         * This why here a first value of the [input] shape is used, which is going to be known at runtime.
         * See also [https://github.com/tensorflow/tensorflow/issues/833](https://github.com/tensorflow/tensorflow/issues/833)
         */
        internal fun Ops.shapeWithDynamicBatchSize(tensorShape: TensorShape, input: Operand<Float>): Operand<Int> {
            val batchSize = squeeze(slice(shape(input), constant(intArrayOf(0)), constant(intArrayOf(1))))
            val otherDims = tensorShape.dims().toList().drop(1).map { constant(it.toInt()) }
            return stack(listOf(batchSize) + otherDims)
        }
    }
}