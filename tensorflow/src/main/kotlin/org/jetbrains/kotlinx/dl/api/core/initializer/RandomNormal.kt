/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors with a normal distribution.
 *
 * @property [mean] Mean of the random values to generate.
 * @property [stdev] Standard deviation of the random values to generate.
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [RandomNormal] initializer.
 */
public class RandomNormal(
    public val mean: Float = 0.0f,
    public val stdev: Float = 1.0f,
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
        val seeds = longArrayOf(seed, 0L)
        val distOp: Operand<Float> = tf.random.statelessRandomNormal(shape, tf.constant(seeds))
        val op: Operand<Float> = tf.math.mul(distOp, tf.constant(stdev))
        return tf.withName(name).math.add(op, tf.constant(mean))
    }

    override fun toString(): String {
        return "RandomNormal(mean=$mean, stdev=$stdev, seed=$seed)"
    }
}
