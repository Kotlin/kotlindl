/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Rectified Linear Unit activation function.
 *
 * With default values, it returns element-wise `max(x, 0)`.
 *
 * Otherwise, it follows:
 * ```
 * f(x) = maxValue,                        if x >= maxValue
 * f(x) = x,                               if threshold <= x < maxValue
 * f(x) = negativeSlope * (x - threshold), if x < threshold
 * ```
 * @property [maxValue] Maximum activation value. Should be >= 0.
 * @property [negativeSlope] Negative slope coefficient. Should be >= 0.
 * @property [threshold] Threshold value for threshold activation.
 * @constructor Creates [ReLU] object.
 * @since 0.2
 */
public class ReLU(
    public val maxValue: Float? = null,
    public val negativeSlope: Float = 0.0f,
    public val threshold: Float = 0.0f,
    name: String = ""
) : AbstractActivationLayer(name) {
    init {
        require(negativeSlope >= 0.0f) { "Negative slope $negativeSlope should be >= 0.0." }
        require(maxValue == null || maxValue >= 0.0f) { "Max value $maxValue should be >= 0.0." }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        // alpha is used for leaky relu slope in activations instead of negativeSlope.
        return commonRelu(tf, input = input, alpha = negativeSlope, maxValue = maxValue, threshold = threshold)
    }

    override fun toString(): String =
        "ReLU(name = $name, maxValue=$maxValue, negativeSlope=$negativeSlope, threshold=$threshold)"
}
