package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

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
 * @property [axis] Axes to apply softmax to (NOTE: the property exist for config compatibility,
 * only the last axis is supported)
 * @constructor Creates [Softmax] object
 * @since 0.3
 */
public class Softmax(
    public val axis: List<Int> = listOf(-1),
    name: String = ""
) : AbstractActivationLayer(name) {
    init {
        if (axis != listOf(-1)) TODO("Handle Softmax layer for non-last axis")
    }

    override fun forward(tf: Ops, input: Operand<Float>): Operand<Float> = tf.nn.softmax(input)

    override fun toString(): String = "Softmax(axis=$axis)"
}