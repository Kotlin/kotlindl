/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.core

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.layer.activation.AbstractActivationLayer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Applies an activation function to an output.
 *
 * @property [activation] Activation function.
 * @property [name] Custom layer name.
 * @constructor Creates [Dense] object.
 *
 * @since 0.2
 */
public class ActivationLayer(
    public val activation: Activations = Activations.Relu,
    name: String = ""
) : AbstractActivationLayer(name) {

    override fun forward(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float> {
        return Activations.convert(activation).apply(tf, input, name)
    }

    override fun toString(): String {
        return "ActivationLayer(activation=$activation)"
    }
}
