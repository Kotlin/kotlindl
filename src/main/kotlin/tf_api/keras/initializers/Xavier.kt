package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.ParameterizedTruncatedNormal
import kotlin.math.sqrt

class Xavier<T : Number>(
    private val seed: Long
) :
    Initializer<T>() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T> {
        val factor = 2.0 // like variance scaling

        val n = (fanIn + fanOut) / 2.0
        val stddev = sqrt(1.3 * factor / n.toFloat()).toFloat()

        println("fanIn $fanIn fanOut $fanOut stddev $stddev")

        return tf.withName(name).random.parameterizedTruncatedNormal(
            shape,
            tf.constant(0f, dtype),
            tf.constant(stddev, dtype),
            tf.constant(-1f, dtype),
            tf.constant(1f, dtype),
            ParameterizedTruncatedNormal.seed(seed)
        )
    }
}