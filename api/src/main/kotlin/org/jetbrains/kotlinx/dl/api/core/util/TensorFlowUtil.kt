/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean

// TODO: rename correctly, move to namespace
/** */
internal fun tfMean(tf: Ops, x: Operand<Float>): Operand<Float> {
    return tfMean(tf, x, null, false)
}

/** */
internal fun tfMean(
    tf: Ops,
    x: Operand<Float>,
    axis: Operand<Int>
): Operand<Float> {
    return tfMean(tf, x, axis, false)
}

/** */
internal fun tfMean(tf: Ops, x: Operand<Float>, keepDims: Boolean): Operand<Float> {
    return tfMean(tf, x, null, keepDims)
}

/** */
internal fun tfMean(
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

