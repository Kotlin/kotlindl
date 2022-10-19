/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Global max pooling operation for 3D data (e.g. videos, spatio-temporal).
 *
 * Downsamples the input by taking the maximum value over spatio-temporal dimensions.
 *
 * @since 0.3
 */
public class GlobalMaxPool3D(
    name: String = "",
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.max(input, tf.constant(intArrayOf(1, 2, 3)))
    }

    override fun toString(): String {
        return "GlobalMaxPool3D(name = $name, hasActivation=$hasActivation)"
    }
}
