package api.keras.initializers

import api.keras.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

/**
 * Initializer that generates a truncated normal distribution.
 *
 * These values are similar to values from a [RandomNormal]
 * except that values more than two standard deviations from the mean are
 * discarded and re-drawn. This is the recommended initializer for neural network
 * weights and filters.
 *
 * @property seed Seed.
 * @constructor Creates [TruncatedNormal] initializer.
 */
class TruncatedNormal(private val seed: Long) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).random.truncatedNormal(
            shape,
            getDType(),
            TruncatedNormal.seed(seed)
        )
    }

    override fun toString(): String {
        return "TruncatedNormal(seed=$seed)"
    }
}