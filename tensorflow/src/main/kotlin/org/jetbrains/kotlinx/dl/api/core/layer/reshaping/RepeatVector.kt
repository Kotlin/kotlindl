/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32

/**
 * Layer that repeats the input [n] times.
 *
 * Input shape: `2D tensor of shape (num_samples, features)`.
 *
 * Output shape: `3D tensor of shape (num_samples, n, features)`.
 *
 * @property n Repetition factor.
 * @property [name] Custom layer name.
 * @constructor Creates [RepeatVector] object.
 *
 * @since 0.3
 */
public class RepeatVector(
    public val n: Int,
    name: String = ""
) : Layer(name) {

    init {
        require(n >= 1) { "Number of repetitions (n) in RepeatVector should be positive but got $n" }
    }

    override fun build(
        tf: Ops,
        input: Operand<TFloat32>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> {
        val x = tf.expandDims(input, tf.constant(1))
        val pattern = tf.stack(listOf(tf.constant(1), tf.constant(n), tf.constant(1)))
        return tf.tile(x, pattern)
    }

    override fun toString(): String {
        return "RepeatVector(name = $name, n=$n, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
