/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Leaky version of a Rectified Linear Unit.
 *
 * It allows a small gradient when the unit is not active:
 * ```
 * f(x) = x,                if x >= 0
 * f(x) = alpha * x         if x < 0
 * ```
 * @property [alpha] Negative slope coefficient. Should be >= 0.
 * @constructor Creates [LeakyReLU] object.
 * @since 0.3
 */
public class LeakyReLU(
    public val alpha: Float = 0.3f,
    name: String = ""
) : AbstractActivationLayer(name) {
    init {
        require(alpha >= 0.0f) {
            "Alpha $alpha should be >= 0.0."
        }
    }

    override fun forward(tf: Ops, input: Operand<Float>): Operand<Float> {
        return commonRelu(tf, input = input, alpha = alpha)
    }

    override fun toString(): String =
        "LeakyReLU(name = $name, alpha=$alpha)"
}
