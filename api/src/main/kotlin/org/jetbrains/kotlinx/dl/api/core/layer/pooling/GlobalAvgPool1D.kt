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
 * NOTE: Works with tensors which must have rank 3 (batch, steps, features).
 * Input shape: 3D tensor with shape `(batch_size, steps, features)`.
 * Output shape: 2D tensor with shape `(batch_size, features)`.
 * @property [name] Custom layer name.
 * @constructor Creates [GlobalAvgPool1D] object.
 */
public class GlobalAvgPool1D(
    name: String = ""
) : Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(inputShape.size(0), inputShape.size(2))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // TODO support for different dataFormat("channel_last", "channel_first")
        val stepAxis = 1
        // TODO support for masking
        return TF.mean(tf, input, tf.constant(stepAxis))
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override fun toString(): String {
        return "GlobalAvgPool1D(name=$name)"
    }
}
