/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors with constant values.
 *
 * @property constantValue Constant value to fill the tensor.
 * @constructor Creates a [Constant] initializer with a given [constantValue].
 */
public class Constant(public val constantValue: Float) : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).fill(shape, tf.constant(constantValue))
    }

    override fun toString(): String {
        return "Constant(constantValue=$constantValue)"
    }
}