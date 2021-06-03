/*
* Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
* Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
*/

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.nn.AvgPool3d

/**
 * Average pooling operation for 3D data (e.g. video, spatio-temporal).
 *
 * Downsamples the input by taking the average over a window of size [poolSize].
 *
 * @property [poolSize] Size of the pooling window.
 * @property [strides] The amount of shift for pooling window in each pooling step. If
 * `null`, it will default to [poolSize].
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no
 * padding, or [ConvPadding.SAME] which means padding the input equally such that the output
 * has the same dimension as the input.
 * @property [dataFormat] Data format of input; can be either of [CHANNELS_LAST] or [CHANNELS_FIRST].
 */
public class AvgPool3D(
    public val poolSize: IntArray = intArrayOf(2, 2, 2),
    public val strides: IntArray? = null,
    public val padding: ConvPadding = ConvPadding.VALID,
    public val dataFormat: String = CHANNELS_LAST,
    name: String = ""
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override val weights: Map<String, Array<*>>
        get() = emptyMap()

    init {
        require(dataFormat == CHANNELS_LAST || dataFormat == CHANNELS_FIRST) {
            "The dataFormat should be either \"$CHANNELS_LAST\" or \"$CHANNELS_FIRST\"."
        }

        require(padding == ConvPadding.VALID || padding == ConvPadding.SAME) {
            "The padding should be either ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }

        require(poolSize.size == 3) {
            "The length of poolSize array should be 3."
        }

        require(strides == null || strides.size == 3) {
            "The strides should be either `null` or an array of length 3."
        }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        val axis1 = if (dataFormat == CHANNELS_LAST) 1 else 2
        var dim1 = inputShape.size(axis1)
        var dim2 = inputShape.size(axis1 + 1)
        var dim3 = inputShape.size(axis1 + 2)
        val strides1 = strides?.get(0) ?: poolSize[0]
        val strides2 = strides?.get(1) ?: poolSize[1]
        val strides3 = strides?.get(2) ?: poolSize[2]
        dim1 = convOutputLength(dim1, poolSize[0], padding, strides1)
        dim2 = convOutputLength(dim2, poolSize[1], padding, strides2)
        dim3 = convOutputLength(dim3, poolSize[3], padding, strides3)

        return if (dataFormat == CHANNELS_LAST) {
            Shape.make(inputShape.size(0), dim1, dim2, dim3, inputShape.size(4))
        } else {
            Shape.make(inputShape.size(0), inputShape.size(1), dim1, dim2, dim3)
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val tfPoolSize = longArrayOf(1, poolSize[0].toLong(), poolSize[1].toLong(), poolSize[2].toLong(), 1)
        val tfStrides = longArrayOf(
            1,
            (strides?.get(0) ?: poolSize[0]).toLong(),
            (strides?.get(1) ?: poolSize[1]).toLong(),
            (strides?.get(2) ?: poolSize[2]).toLong(),
            1
        )
        val tfPadding = padding.paddingName
        val tfDataFormat = if (dataFormat == CHANNELS_LAST) "NDHWC" else "NCDHW"
        return tf.nn.avgPool3d(
            input,
            tfPoolSize.toList(),
            tfStrides.toList(),
            tfPadding,
            AvgPool3d.dataFormat(tfDataFormat)
        )
    }

    override fun toString(): String =
        "AvgPool3D(poolSize=$poolSize, strides=$strides, padding=$padding, dataFormat=$dataFormat)"
}
