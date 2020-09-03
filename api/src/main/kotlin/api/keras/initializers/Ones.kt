package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Ones : Initializer() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).fill(shape, tf.constant(1.0f))
    }
}