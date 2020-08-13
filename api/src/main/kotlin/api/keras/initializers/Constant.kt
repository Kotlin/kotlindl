package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Constant<T : Number>(private val constantValue: Any) : Initializer<T>() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T> {
        return tf.withName(name).fill(shape, tf.constant(constantValue, dtype))
    }
}