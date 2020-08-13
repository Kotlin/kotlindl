package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

class TruncatedNormal<T : Number>(private val seed: Long) :
    Initializer<T>() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T> {
        return tf.withName(name).random.truncatedNormal(
            shape,
            dtype,
            TruncatedNormal.seed(seed)
        )
    }
}