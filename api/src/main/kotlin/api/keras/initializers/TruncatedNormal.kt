package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

class TruncatedNormal(private val seed: Long) :
    Initializer() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<Float>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).random.truncatedNormal(
            shape,
            dtype,
            TruncatedNormal.seed(seed)
        )
    }
}