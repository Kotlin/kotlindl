/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Layer that subtracts inputs, following from the previous.
 *
 * It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape),
 * (```inputs[i-1] - inputs[i]```), also of the same shape.
 */
public class Subtract(name: String = "") : AbstractMerge("SubtractLayer", name) {
    override fun mergeFunction(
        input: List<Operand<Float>>,
        tf: Ops
    ): Operand<Float> {
        var output = input[0]
        for (i in 1 until input.size)
            output = tf.math.sub(output, input[i])
        return output
    }

    override fun toString(): String {
        return "Subtract(name = $name)"
    }
}
