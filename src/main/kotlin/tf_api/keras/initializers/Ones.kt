package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Ones<T : Number> : Initializer<T>() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T> {
        return tf.withName(name).fill(shape, tf.constant(1.0f, dtype))
    }
}