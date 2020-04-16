package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Zeros : Initializer() {
    override fun <T : Number> initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T> {
        return tf.zeros(shape, dtype)
    }
}
