/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Rectified linear unit.
 *
 * With default values, it returns element-wise `max(x, 0)`.
 * Otherwise, it follows:
 * `f(x) = max_value` for `x >= max_value`,
 * `f(x) = x` for `threshold <= x < max_value`,
 * `f(x) = alpha * (x - threshold)` otherwise.
 *
 * @param [tf] Namespace to build ops.
 * @param [input] A tensor or variable.
 * @param [alpha] A scalar, slope of negative section.
 * @param [maxValue] Saturation threshold.
 * @param [threshold] Threshold value for the activation.
 *
 * @return TensorFlow Operand.
 */
internal fun commonRelu(
    tf: Ops,
    input: Operand<Float>,
    alpha: Float = 0.0f,
    maxValue: Float? = null,
    threshold: Float = 0.0f
): Operand<Float> {
    var input2 = input
    var negativePart: Operand<Float> = tf.nn.relu(input2) // fake init
    if (alpha != 0.0f) {
        if (maxValue == null && threshold == 0.0f) {
            // LeakyReLU
            val greaterThanZero = tf.math.greater(input2, tf.constant(0.0f))
            val negativeActivation = tf.math.mul(tf.constant(alpha), input2)
            return tf.where3(greaterThanZero, input2, negativeActivation)
        }
        negativePart = if (threshold != 0.0f)
            tf.nn.relu(tf.math.add(tf.math.mul(input2, tf.constant(-1.0f)), tf.constant(threshold)))
        else
            tf.nn.relu(tf.math.mul(input2, tf.constant(-1.0f)))
    }

    var clipMax = false
    if (maxValue != null) clipMax = true

    when {
        threshold != 0.0f -> {
            input2 = tf.math.mul(input2, tf.dtypes.cast(tf.math.greater(input, tf.constant(threshold)), getDType()))
        }
        maxValue == 6.0f -> {
            input2 = tf.nn.relu6(input2)
            clipMax = false
        }
        else -> input2 = tf.nn.relu(input2)
    }

    if (clipMax) {
        input2 = tf.math.minimum(tf.constant(maxValue!!) as Operand<Float>, tf.math.maximum(input2, tf.constant(0.0f)))
        // original code replaced on composition of min and max calls due to missed gradients for clipByValue op
        /*tf.clipByValue(
                input2,
                tf.constant(0.0f) as Operand<Float>,
                tf.constant(maxValue!!)
            )*/
    }

    if (alpha != 0.0f) input2 = tf.math.sub(input2, tf.math.mul(tf.constant(alpha), negativePart))

    return input2
}
