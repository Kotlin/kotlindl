/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

/**
 * Average pooling operation for 1D temporal data (e.g. audio, time-series).
 *
 * Downsamples the input by taking the average over a temporal window of size [poolSize].
 *
 * @property [poolSize] Size of the temporal pooling window for each dimension of input.
 * @property [strides] The amount of shift for pooling window per each input dimension in each pooling step.
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no
 * padding, or [ConvPadding.SAME] which means padding the input equally such that the output
 * has the same dimension as the input.
 *
 * @since 0.3
 */
public class AvgPool1D(
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
            "The padding should be either of ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }
    }

    override fun build(tf: Ops, inputShape: Shape): Shape {
        var steps = inputShape.size(1)
        steps = convOutputLength(steps, poolSize[1], padding, strides[1])
        return Shape.make(inputShape.size(0), steps, inputShape.size(2))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val expandAxis = 2
        val tfInput = tf.expandDims(input, tf.constant(expandAxis))

        val tfPoolSize = longArrayOf(1, 1, 1, 1)
        val tfStrides = longArrayOf(1, 1, 1, 1)
        tfPoolSize[expandAxis - 1] = poolSize[1].toLong()
        tfStrides[expandAxis - 1] = strides[1].toLong()

        val tfPadding = padding.paddingName

        val avgPool = tf.nn.avgPool(
            tfInput,
            tfPoolSize.toList(),
            tfStrides.toList(),
            tfPadding
        )

        return tf.squeeze(avgPool, Squeeze.axis(listOf(expandAxis.toLong())))
    }

    override fun toString(): String {
        return "AvgPool1D(name = $name, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }
}
