package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.ParameterizedTruncatedNormal

class RandomNormal<T : Number>(
    private val mean: Float,
    private val stdev: Float,
    private val p1: Float,
    private val p2: Float,
    private val seed: Long
) :
    Initializer<T>() {
    override fun initialize(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T> {
        return tf.withName(name).random.parameterizedTruncatedNormal(
            shape,
            tf.constant(mean, dtype),
            tf.constant(stdev, dtype),
            tf.constant(p1, dtype),
            tf.constant(p2, dtype),
            ParameterizedTruncatedNormal.seed(seed)
        )
    }
}