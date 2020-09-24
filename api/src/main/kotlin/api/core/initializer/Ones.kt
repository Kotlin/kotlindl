package api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors initialized to 1.
 */
class Ones : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).fill(shape, tf.constant(1.0f))
    }

    override fun toString(): String {
        return "Ones()"
    }
}