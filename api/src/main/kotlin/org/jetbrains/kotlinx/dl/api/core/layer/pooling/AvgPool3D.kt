/*
* Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
* Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
*/

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.requireArraySize
import org.jetbrains.kotlinx.dl.api.core.layer.toLongList
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.tensorflow.Operand
import org.tensorflow.Shape
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
 */
public class AvgPool3D(
    public val poolSize: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val strides: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    init {
        requireArraySize(poolSize, 5, "poolSize")
        requireArraySize(strides, 5, "strides")
        require(padding == ConvPadding.VALID || padding == ConvPadding.SAME) {
            "The padding should be either ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape): Unit = Unit

    override fun computeOutputShape(inputShape: Shape): Shape {
        val dim1 = convOutputLength(inputShape.size(1), poolSize[1], padding, strides[1])
        val dim2 = convOutputLength(inputShape.size(2), poolSize[2], padding, strides[2])
        val dim3 = convOutputLength(inputShape.size(3), poolSize[3], padding, strides[3])

        return Shape.make(inputShape.size(0), dim1, dim2, dim3, inputShape.size(4))
    }

    override fun forward(
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
        return "AvgPool3D(name = $name, isTrainable=$isTrainable, poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding, hasActivation=$hasActivation)"
    }
}
