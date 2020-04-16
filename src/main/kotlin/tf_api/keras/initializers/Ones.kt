package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Ones : Initializer() {
    override fun <T : Number> initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T> {
        return tf.fill(shape, tf.constant(1.0f, dtype))
    }
}