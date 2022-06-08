/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Exponential Unit activation function.
 *
 * It follows:
 * ```
 * f(x) = x,                    if x > 0
 * f(x) = alpha * (exp(x) - 1), if x <= 0
 * ```
 *
 * In contrast to ReLU it has negative values which push the mean of
 * the activation closer to zero which enable faster learning as they
 * bring the gradient to the natural gradient.
 *
 * @property [alpha] Hyperparameter that controls the value to which
 * an ELU saturates for negative net inputs. Should be > 0.
 * @constructor Creates [ELU] object.
 * @since 0.3
 */
public class ELU(
    public val alpha: Float = 1.0f,
    name: String = ""
) : AbstractActivationLayer(name) {
    init {
        require(alpha > 0.0f) { "Alpha $alpha should be > 0.0." }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> = when (alpha) {
        1.0f -> tf.nn.elu(input)
        else -> {
            val greaterThanZero = tf.math.greater(input, tf.constant(0.0f))
            val scaledActivation = tf.math.mul(tf.constant(alpha), tf.nn.elu(input))
            tf.where3(greaterThanZero, input, scaledActivation)
        }
    }

    override fun toString(): String =
        "ELU(name = $name, alpha=$alpha)"
}
