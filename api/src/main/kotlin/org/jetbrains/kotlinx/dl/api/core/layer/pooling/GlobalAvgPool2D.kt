/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.TF
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Global average pooling operation for 2D data (images and so on).
 *
 * NOTE: Only works with tensors which have rank 4, i.e. tensors with shape
 * `(batch, rows, columns, channels)` or `(batch, channels, rows, columns)`.
 *
 * @property [dataFormat] Data format of input; can be either of [CHANNELS_LAST] or [CHANNELS_FIRST].
 * @constructor Creates [GlobalAvgPool2D] object.
 *
 * @since 0.2
 */
public class GlobalAvgPool2D(
    public val dataFormat: String = CHANNELS_LAST,
    name: String = ""
) : Layer(name) {
    init {
        require(dataFormat == CHANNELS_LAST || dataFormat == CHANNELS_FIRST) {
            "The dataFormat should be either of \"$CHANNELS_LAST\" or \"$CHANNELS_FIRST\"."
        }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return if (dataFormat == CHANNELS_LAST) {
            Shape.make(inputShape.size(0), inputShape.size(3))
        } else {
            Shape.make(inputShape.size(0), inputShape.size(1))
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val spatialAxes = if (dataFormat == CHANNELS_LAST) intArrayOf(1, 2) else intArrayOf(2, 3)
        return TF.mean(tf, input, tf.constant(spatialAxes))
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "GlobalAvgPool2D(dataFormat=$dataFormat)"
    }
}
