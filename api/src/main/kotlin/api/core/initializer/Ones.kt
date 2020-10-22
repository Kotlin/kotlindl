/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors initialized to 1.
 *
 * NOTE: It does not work properly during model import/export, known issue: https://github.com/zaleslaw/Kotof/issues/4.
 */
public class Ones : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).fill(shape, tf.constant(1.0f))
    }

    override fun toString(): String {
        return "Ones()"
    }
}