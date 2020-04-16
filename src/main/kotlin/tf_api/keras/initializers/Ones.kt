package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Ones<T : Number> : Initializer<T>() {
    override fun initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T> {
        return tf.fill(shape, tf.constant(1.0f, dtype))
    }
}