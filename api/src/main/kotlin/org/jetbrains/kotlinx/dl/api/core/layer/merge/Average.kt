/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Layer that averages a list of inputs element-wise.
 *
 * It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape).
 */
public class Average(name: String = "") : AbstractMerge("AverageLayer", name) {
    override fun mergeFunction(
        input: List<Operand<Float>>,
        tf: Ops
    ): Operand<Float> {
        var output = input[0]
        for (i in 1 until input.size)
            output = tf.math.add(output, input[i])

        output = tf.math.div(output, tf.constant(input.size.toFloat()))
        return output
    }

    override fun toString(): String {
        return "Average(name = $name, isTrainable=$isTrainable,)"
    }
}
