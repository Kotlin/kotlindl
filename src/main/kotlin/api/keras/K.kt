package api.keras

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.math.Mean

fun <T : Number> Kmean(tf: Ops, x: Operand<T>): Operand<T> {
    return Kmean(tf, x, null, false)
}

fun <T : Number> Kmean(
    tf: Ops,
    x: Operand<T>,
    axis: Operand<Int>
): Operand<T> {
    return Kmean(tf, x, axis, false)
}

fun <T : Number> Kmean(tf: Ops, x: Operand<T>, keepDims: Boolean): Operand<T> {
    return Kmean(tf, x, null, keepDims)
}

fun <T : Number> Kmean(
    tf: Ops,
    x: Operand<T>,
    axis: Operand<Int>?,
    keepDims: Boolean
): Operand<T> {
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