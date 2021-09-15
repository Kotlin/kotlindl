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
 * Global Average pooling operation for 3D data.
 *
 * NOTE: Works with tensors which must have rank 5 (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels).
 *
 * Input shape: 5D tensor with shape `(batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)`.
 * Output shape: 2D tensor with shape `(batch_size, channels)`.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [GlobalAvgPool3D] object.
 */
public class GlobalAvgPool3D(
    name: String = ""
) : Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        // TODO add dataFormat support
        return Shape.make(inputShape.size(0), inputShape.size(4))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // TODO add dataFormat support
        return TF.mean(tf, input, tf.constant(intArrayOf(1, 2, 3)))
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override fun toString(): String {
        return "GlobalAvgPool3D(name=$name)"
    }
}
