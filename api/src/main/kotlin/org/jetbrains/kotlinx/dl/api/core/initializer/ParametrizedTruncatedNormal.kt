/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.ParameterizedTruncatedNormal

/**
 * Initializer that generates a parametrized truncated normal distribution.
 *
 * @property [mean] Mean of the random values to generate.
 * @property [stdev] Standard deviation of the random values to generate.
 * @property [p1] The minimum cutoff. May be -infinity.
 * @property [p2] The maximum cutoff. May be +infinity, and must be more than the minval
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [ParametrizedTruncatedNormal] initializer.
 */
public class ParametrizedTruncatedNormal(
    internal val mean: Float = 0.0f,
    internal val stdev: Float = 1.0f,
    internal val p1: Float = -10.0f, // low level edge
    internal val p2: Float = 10.0f, // high level edge
    internal val seed: Long
) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        require(p1 < p2) { "The p1 parameter value must be less than p2 parameter value." }

        return tf.withName(name).random.parameterizedTruncatedNormal(
            shape,
            tf.constant(mean),
            tf.constant(stdev),
            tf.constant(p1),
            tf.constant(p2),
            ParameterizedTruncatedNormal.seed(seed)
        )
    }

    override fun toString(): String {
        return "ParametrizedTruncatedNormal(mean=$mean, stdev=$stdev, p1=$p1, p2=$p2, seed=$seed)"
    }
}
