package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Constant<T : Number>(private val constantValue: Any) : Initializer<T>() {
    override fun initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T> {
        return tf.fill(shape, tf.constant(constantValue, dtype))
    }

}