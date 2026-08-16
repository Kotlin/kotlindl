/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32

/** Helper class that emulates Keras functions from tensorflow.keras. */
public object TF {
    /** */
    internal fun mean(tf: Ops, x: Operand<TFloat32>): Operand<TFloat32> {
        return mean(tf, x, null, false)
    }

    /** */
    internal fun mean(
        tf: Ops,
        x: Operand<TFloat32>,
        axis: Operand<TInt32>
    ): Operand<TFloat32> {
        return mean(tf, x, axis, false)
    }

    /** */
    internal fun mean(tf: Ops, x: Operand<TFloat32>, keepDims: Boolean): Operand<TFloat32> {
        return mean(tf, x, null, keepDims)
    }

    /** */
    internal fun mean(
        tf: Ops,
        x: Operand<TFloat32>,
        axis: Operand<TInt32>?,
        keepDims: Boolean
    ): Operand<TFloat32> {
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
