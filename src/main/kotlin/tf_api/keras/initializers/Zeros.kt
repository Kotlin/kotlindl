package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Zeros<T : Number> : Initializer<T>() {
    override fun initialize(funIn: Int, funOut: Int, tf: Ops, shape: Operand<Int>, dtype: Class<T>): Operand<T> {
        return tf.zeros(shape, dtype)
    }
}
