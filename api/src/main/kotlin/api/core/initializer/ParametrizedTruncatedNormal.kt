package api.core.initializer

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.ParameterizedTruncatedNormal

/**
 *
 */
class ParametrizedTruncatedNormal(
    private val mean: Float = 0.0f,
    private val stdev: Float = 1.0f,
    private val p1: Float = -10.0f, // low level edge
    private val p2: Float = 10.0f, // high level edge
    private val seed: Long
) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        require(p1 < p2) { "The p1 parameter value must be less than p2 parameter value." }

        return tf.withName(name).random.parameterizedTruncatedNormal(
            shape,
            tf.constant(mean),
            tf.constant(stdev),
            tf.constant(p1),
            tf.constant(p2),
            ParameterizedTruncatedNormal.seed(seed)
        )
    }

    override fun toString(): String {
        return "ParametrizedTruncatedNormal(mean=$mean, stdev=$stdev, p1=$p1, p2=$p2, seed=$seed)"
    }
}