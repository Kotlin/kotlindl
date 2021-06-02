/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Global max pooling operation for 1D temporal data (e.g. audio, timseries, etc.).
 *
 * Downsamples the input by taking the maximum value over time dimension.
 *
 * @property [dataFormat] The order of dimensions in the input.
 */
public class GlobalMaxPool1D(
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
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return if (dataFormat == CHANNELS_LAST) {
            Shape.make(inputShape.size(0), inputShape.size(2))
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
        return if (dataFormat == CHANNELS_LAST) {
            tf.max(input, tf.constant(1))
        } else {
            tf.max(input, tf.constant(2))
        }
    }

    override fun toString(): String =
        "GlobalMaxPool1D(dataFormat=$dataFormat)"
}
