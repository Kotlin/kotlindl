package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.tensorflow.Operand
import org.tensorflow.op.Ops

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