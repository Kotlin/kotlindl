/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

/**
 * Max pooling operation for 1D temporal data (e.g. audio, timeseries).
 *
 * Downsamples the input by taking maximum value over a temporal window of size [poolSize].
 *
 * @property [poolSize] Size of the temporal pooling window for each dimension of input.
 * @property [strides] The amount of shift for pooling window per each input dimension in each pooling step.
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no padding, or
 * [ConvPadding.SAME] which means padding the input equally such that the output has the same dimension
 * as the input.
 *
 * @since 0.3
 */
public class MaxPool1D(
    public val poolSize: IntArray = intArrayOf(1, 2, 1),
    public val strides: IntArray = intArrayOf(1, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {
    public constructor(
        poolSize: Int = 2,
        strides: Int = 2,
        padding: ConvPadding = ConvPadding.VALID,
        name: String = ""
    ) : this(
        poolSize = intArrayOf(1, poolSize, 1),
        strides = intArrayOf(1, strides, 1),
        padding = padding,
        name = name
    )

    override val hasActivation: Boolean
        get() = false

    init {
        requireArraySize(poolSize, 3, "poolSize")
        requireArraySize(strides, 3, "strides")
        require(padding == ConvPadding.VALID || padding == ConvPadding.SAME) {
            "The padding should be either ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }
    }

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        /**
         * Since the low-level Java API does not provide a function for 1D max-pooling,
         * the 2D pooling should be used instead; therefore, the input is expanded first
         * and an equivalent 2D pooling filter is constructed and used to achieve the same
         * effect.
         */
        val expandAxis = 2
        val tfInput = tf.expandDims(input, tf.constant(expandAxis))
        val tfPoolSize = intArrayOf(1, 1, 1, 1)
        val tfStrides = intArrayOf(1, 1, 1, 1)
        /**
         * NOTE: we can use `MaxPool.Options` argument of `tf.nn.maxPool` to pass
         * the data format, as follows:
         * ```
         * val tfDataFormat = if (dataFormat == CHANNELS_LAST) "NHWC" else "NCHW_VECT_C"
         * tf.nn.maxPool(..., MaxPool.dataFormat(tfDataFormat))
         * ```
         * However, it seems it does not work for the case of "channels_first". So, instead
         * we are choosing to set the value of pool size and strides based on the data format.
         */
        tfPoolSize[expandAxis - 1] = poolSize[1]
        tfStrides[expandAxis - 1] = strides[1]

        val maxPool = tf.nn.maxPool(
            tfInput,
            tf.constant(tfPoolSize),
            tf.constant(tfStrides),
            padding.paddingName
        )
        return tf.squeeze(maxPool, Squeeze.axis(listOf(expandAxis.toLong())))
    }

    override fun toString(): String {
        return "MaxPool1D(name = $name, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }
}
