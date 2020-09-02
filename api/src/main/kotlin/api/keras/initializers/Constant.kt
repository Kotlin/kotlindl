package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Constant(private val constantValue: Any) : Initializer() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<Float>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).fill(shape, tf.constant(constantValue, dtype))
    }
}