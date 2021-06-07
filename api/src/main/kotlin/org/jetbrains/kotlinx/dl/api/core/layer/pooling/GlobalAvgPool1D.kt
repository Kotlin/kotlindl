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
 * Global average pooling operation for temporal data.
 *
 * NOTE: Only works with tensors which have rank 3, i.e. tensors with shape
 * `(batch, steps, features)`.
 *
 * @constructor Creates [GlobalAvgPool1D] object.
 */
public class GlobalAvgPool1D(
    name: String = ""
) : Layer(name) {
    // TODO: add support for `dataFormat` (i.e. "channels_last" or "channels_first"

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) { }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(inputShape.size(0), inputShape.size(2))
    }

    override fun forward(tf: Ops, input: Operand<Float>, isTraining: Operand<Boolean>, numberOfLosses: Operand<Float>?): Operand<Float> {
        val stepsAxis = 1
        // TODO support for masking
        return TF.mean(tf, input, tf.constant(stepsAxis))
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "GlobalAvgPool1D()"
    }
}
