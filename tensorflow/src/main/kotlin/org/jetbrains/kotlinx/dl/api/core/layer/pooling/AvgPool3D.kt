/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.util.toLongList
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Average pooling operation for 3D data (e.g. video, spatio-temporal).
 *
 * Downsamples the input by taking the average over a window of size [poolSize].
 *
 * @property [poolSize] Size of the pooling window for each dimension of input.
 * @property [strides] The amount of shift for pooling window per each input dimension in each pooling step.
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no
 * padding, or [ConvPadding.SAME] which means padding the input equally such that the output
 * has the same dimension as the input.
 *
 * @since 0.3
 */
public class AvgPool3D(
    public val poolSize: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val strides: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {
    public constructor(
        poolSize: Int = 2,
        strides: Int = 2,
        padding: ConvPadding = ConvPadding.VALID,
        name: String = ""
    ) : this(
        poolSize = intArrayOf(1, poolSize, poolSize, poolSize, 1),
        strides = intArrayOf(1, strides, strides, strides, 1),
        padding = padding,
        name = name
    )

    override val hasActivation: Boolean
        get() = false

    init {
        requireArraySize(poolSize, 5, "poolSize")
        requireArraySize(strides, 5, "strides")
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
        return tf.nn.avgPool3d(
            input,
            poolSize.toLongList(),
            strides.toLongList(),
            padding.paddingName
        )
    }

    override fun toString(): String {
        return "AvgPool3D(name = $name, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }
}
