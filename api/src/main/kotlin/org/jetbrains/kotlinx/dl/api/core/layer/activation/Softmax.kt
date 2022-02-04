package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.ReduceSum

import org.tensorflow.op.core.ReduceMax

/**
 * Softmax activation layer
 *
 * Rescales input such that the elements in [axis] sum up to 1.
 *
 * For each batch `i` and class `j` we have
 *
 * ```
 * softmax[i, j] = exp(logits[i, j]) / sum_j(exp(logits[i, j]))
 * ```
 *
 * @property [axis] along which the softmax normalization is applied.
 * @constructor Creates [Softmax] object
 * @since 0.3
 */
public class Softmax(
    public val axis: List<Int> = listOf(-1),
    name: String = ""
) : AbstractActivationLayer(name) {
    init {
        if (axis.size != 1) throw Exception("Multiple axes are not supported")
    }

    override fun forward(tf: Ops, input: Operand<Float>): Operand<Float> {
        val shape = tf.shape(input)
        val numDimensions = tf.size(shape)
        return if (numDimensions == tf.constant(2)) {
            tf.nn.softmax(input)
        } else {
            val e: Operand<Float> = tf.math.exp(
                tf.math.sub(input, tf.reduceMax(input, tf.constant(axis.first()), ReduceMax.keepDims(true)))
            )
            val s: Operand<Float> = tf.reduceSum(e, tf.constant(axis.first()), ReduceSum.keepDims(true))
            tf.math.div(e, s)
        }
    }

    override fun toString(): String {
        return "Softmax(name = $name, isTrainable=$isTrainable, axis=$axis)"
    }
}
