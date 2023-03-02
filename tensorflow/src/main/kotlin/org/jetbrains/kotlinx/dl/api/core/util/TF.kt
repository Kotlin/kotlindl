/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean

/** Helper class that emulates Keras functions from tensorflow.keras. */
public object TF {
    /** */
    internal fun mean(tf: Ops, x: Operand<Float>): Operand<Float> {
        return mean(tf, x, null, false)
    }

    /** */
    internal fun mean(
        tf: Ops,
        x: Operand<Float>,
        axis: Operand<Int>
    ): Operand<Float> {
        return mean(tf, x, axis, false)
    }

    /** */
    internal fun mean(tf: Ops, x: Operand<Float>, keepDims: Boolean): Operand<Float> {
        return mean(tf, x, null, keepDims)
    }

    /** */
    internal fun mean(
        tf: Ops,
        x: Operand<Float>,
        axis: Operand<Int>?,
        keepDims: Boolean
    ): Operand<Float> {
        var localAxis = axis

        if (localAxis == null) {
            val rank: Int = x.asOutput().shape().numDimensions()
            val ranks = IntArray(rank)
            for (i in 0 until rank) {
                ranks[i] = i
            }
            localAxis = tf.constant(ranks)
        }
        return tf.math.mean(x, localAxis, Mean.keepDims(keepDims))
    }
}
