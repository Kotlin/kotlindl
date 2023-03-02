/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

/**
 * Initializer that generates a truncated normal distribution.
 *
 * These values are similar to values from a [RandomNormal]
 * except that values more than two standard deviations from the mean are
 * discarded and re-drawn. This is the recommended initializer for neural network
 * weights and filters.
 *
 * @property seed Seed.
 * @constructor Creates [TruncatedNormal] initializer.
 */
public class TruncatedNormal(public val seed: Long = 12L) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).random.truncatedNormal(
            shape,
            getDType(),
            TruncatedNormal.seed(seed)
        )
    }

    override fun toString(): String {
        return "TruncatedNormal(seed=$seed)"
    }
}