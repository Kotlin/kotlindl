package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

class TruncatedNormal<T : Number>(private val seed: Long) :
    Initializer<T>() {
    override fun initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T> {


        val truncatedNormal = tf.random.truncatedNormal(
            shape,
            dtype,
            TruncatedNormal.seed(seed)
        )
        return tf.math.mul(truncatedNormal, tf.constant(0.1f) as Operand<T>)
    }
}