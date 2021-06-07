/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.TF
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Global average pooling operation for 2D data (images and so on).
 *
 * NOTE: Only works with tensors which have rank 4, i.e. tensors with shape
 * `(batch, rows, columns, channels)`.
 *
 * @constructor Creates [GlobalAvgPool2D] object.
 *
 * @since 0.2
 */
public class GlobalAvgPool2D(
    name: String = ""
) : Layer(name) {
    // TODO: add support for `dataFormat` (i.e. "channels_last" or "channels_first")

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(inputShape.size(0), inputShape.size(3))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val spatialAxes = intArrayOf(1, 2)
        return TF.mean(tf, input, tf.constant(spatialAxes))
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "GlobalAvgPool2D()"
    }
}
