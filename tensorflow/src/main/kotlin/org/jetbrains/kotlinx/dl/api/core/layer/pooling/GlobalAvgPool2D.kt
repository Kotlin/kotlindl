/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.TF
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Global average pooling operation for 2D data (images and so on).
 *
 * NOTE: Works with tensors which must have rank 4 (batch, height, width, channels).
 *
 * Input shape: 4D tensor with shape `(batch_size, rows, cols, channels)`.
 *
 * Output shape: 2D tensor with shape `(batch_size, channels)`.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [GlobalAvgPool2D] object.
 *
 * @since 0.2
 */
public class GlobalAvgPool2D(
    name: String = ""
) : Layer(name) {
    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return TF.mean(tf, input, tf.constant(intArrayOf(1, 2)))
    }

    override fun toString(): String {
        return "GlobalAvgPool2D(name = $name, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
