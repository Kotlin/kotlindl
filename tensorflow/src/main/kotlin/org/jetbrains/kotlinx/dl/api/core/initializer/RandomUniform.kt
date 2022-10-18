/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * @property [minVal] Lower bound of the range of random values to generate (inclusive).
 * @property [maxVal] Upper bound of the range of random values to generate (exclusive).
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [RandomUniform] initializer.
 */
public class RandomUniform(
    public val maxVal: Float = 1.0f,
    public val minVal: Float = 0.0f,
    public val seed: Long = 12L
) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        require(minVal <= maxVal) { "The minVal parameter value must be less or equal than maxVal parameter value." }

        val seeds = longArrayOf(seed, 0L)
        var distOp: Operand<Float> = tf.random.statelessRandomUniform(shape, tf.constant(seeds), getDType())
        if (minVal == 0.0f) {
            if (maxVal != 1.0f) {
                distOp = tf.math.mul(distOp, tf.constant(maxVal))
            }
        } else {
            distOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(maxVal - minVal), getDType()))
            distOp = tf.math.add(distOp, tf.dtypes.cast(tf.constant(minVal), getDType()))
        }

        return tf.withName(name).identity(distOp)
    }

    override fun toString(): String {
        return "RandomUniform(maxVal=$maxVal, minVal=$minVal, seed=$seed)"
    }
}
