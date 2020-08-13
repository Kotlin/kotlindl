package api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.ParameterizedTruncatedNormal

class ParametrizedTruncatedNormal<T : Number>(
    private val mean: Float,
    private val stdev: Float,
    private val p1: Float, // low level edge
    private val p2: Float, // high level edge
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
        require(p1 < p2) { "The p1 parameter value must be less than p2 parameter value." }

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