/*
 * Copyright 2020-2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.regularizer

import org.jetbrains.kotlinx.dl.api.core.loss.allAxes
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/** Default penalty. */
public const val DEFAULT_PENALTY: Float = 0.00001f

/**
 * A regularizer that applies both L1 and L2 regularization penalties.
 * The L1 regularization penalty is computed as:
 *
 * ```loss = l1 * reduce_sum(abs(x))```
 *
 * The L2 regularization penalty is computed as
 *
 * ```loss = l2 * reduce_sum(square(x))```
 *
 * This is a common class which have two specific implementations: [L1] and [L2].
 */
public open class L2L1(
    /**
     * Value of L1 regularization penalty. Could be calculated during graph computations.
     */
    public val l1: Float = DEFAULT_PENALTY,
    /**
     * Value of L2 regularization penalty. Could be calculated during graph computations.
     */
    public val l2: Float = DEFAULT_PENALTY
) : Regularizer() {
    override fun apply(tf: Ops, input: Operand<Float>): Operand<Float> {
        var regularization: Operand<Float> = tf.constant(0.0f)
        if (l1 == 0f && l2 == 0f) {
            return regularization
        } else {
            if (l1 != 0f) {
                val reduceSum = tf.reduceSum(tf.math.abs(input), allAxes(tf, input))
                regularization = tf.math.add(regularization, tf.math.mul(tf.constant(l1), reduceSum))
            }

            if (l2 != 0f) {
                // used math.mul instead square due to Gradient crashes
                val reduceSum = tf.reduceSum(tf.math.mul(input, input), allAxes(tf, input))
                regularization = tf.math.add(regularization, tf.math.mul(tf.constant(l2), reduceSum))
            }
        }
        return regularization
    }

    override fun toString(): String {
        return "L2L1(l1=$l1, l2=$l2)"
    }
}

/**
 * A regularizer that applies a L1 regularization penalty.
 *
 * The L1 regularization penalty is computed as:
 *
 * ```loss = l1 * reduceSum(abs(x))```
 */
public class L1(value: Float = DEFAULT_PENALTY) : L2L1(l1 = value)

/**
 * A regularizer that applies a L2 regularization penalty.
 *
 * The L2 regularization penalty is computed as:
 *
 * ```loss = l2 * reduce_sum(square(x))```
 */
public class L2(value: Float = DEFAULT_PENALTY) : L2L1(l2 = value)
