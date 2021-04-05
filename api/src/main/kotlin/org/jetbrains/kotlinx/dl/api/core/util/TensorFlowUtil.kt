/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean
import java.nio.Buffer
import java.nio.FloatBuffer

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

/** Converts [src] to [FloatBuffer] from [start] position for the next [length] positions. */
public fun serializeToBuffer(src: Array<FloatArray>, start: Int, length: Int): FloatBuffer {
    val buffer = FloatBuffer.allocate(length * src[0].size)
    for (i in start until start + length) {
        buffer.put(src[i])
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

// TODO: converts all given data to FloatBuffer
// TODO: use this function to preprocess on Tensor.create everywhere it's possible
/** Converts [src] to [FloatBuffer]. */
public fun serializeToBuffer(src: Array<FloatArray>): FloatBuffer {
    val buffer = FloatBuffer.allocate(src.size * src[0].size)
    for (element in src) {
        buffer.put(element)
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

/** Converts [src] to [FloatBuffer]. */
public fun serializeToBuffer(src: FloatArray): FloatBuffer {
    val buffer = FloatBuffer.allocate(src.size)
    buffer.put(src)
    return (buffer as Buffer).rewind() as FloatBuffer
}

/** Converts [src] to [FloatBuffer]. */
public fun serializeLabelsToBuffer(src: FloatArray, amountOfClasses: Long): FloatBuffer {
    val oneHotEncodedLabels = Array(src.size) {
        FloatArray(amountOfClasses.toInt()) { 0.0f }
    }
    for (i in src.indices) {
        val label = src[i]
        val labelToIndexMapping = if (amountOfClasses == 1L) {
            0 // for regression tasks
        } else label.toInt() //TODO: should be implemented with the real OHE here
        // TODO: classes should be numerated from [0.0 to amountOfClasses), need a require check on input data
        oneHotEncodedLabels[i][labelToIndexMapping] = 1f // TODO: check that this is work for regression
    }

    val buffer = FloatBuffer.allocate(oneHotEncodedLabels.size * oneHotEncodedLabels[0].size)
    for (element in oneHotEncodedLabels) {
        buffer.put(element)
    }
    return (buffer as Buffer).rewind() as FloatBuffer
}

