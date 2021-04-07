package org.jetbrains.kotlinx.dl.api.core.layer.pooling

/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.tfMean
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Average pooling layer for 2D inputs (e.g. images).
 *
 * NOTE: Works with tensors which must have rank 4 (batch, height, width, channels).
 *
 * @property [poolSize] The size of the sliding window for each dimension of input tensor (pool batch, pool height, pool width, pool channels).
 * Usually, pool batch and pool channels are equal to 1.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @constructor Creates [GlovalAvgPool2D] object.
 */
public class GlobalAvgPool2D(
    name: String = ""
) : Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(inputShape.size(0), inputShape.size(3)) //   if (this.dataFormat == 'channelsLast')
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tfMean(tf, input, tf.constant(intArrayOf(1, 2)))
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "GlobalAvgPool2D()"
    }
}
