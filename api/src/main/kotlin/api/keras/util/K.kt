package api.keras.util

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean

fun Kmean(tf: Ops, x: Operand<Float>): Operand<Float> {
    return Kmean(tf, x, null, false)
}

fun Kmean(
    tf: Ops,
    x: Operand<Float>,
    axis: Operand<Int>
): Operand<Float> {
    return Kmean(tf, x, axis, false)
}

fun Kmean(tf: Ops, x: Operand<Float>, keepDims: Boolean): Operand<Float> {
    return Kmean(tf, x, null, keepDims)
}

fun Kmean(
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