package api.core.initializer

import api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates tensors with a uniform distribution.
 *
 * @property [maxVal] Lower bound of the range of random values to generate (inclusive).
 * @property [minVal] Upper bound of the range of random values to generate (exclusive).
 * @property [seed] Used to create random seeds.
 * @constructor Creates a [RandomUniform] initializer.
 */
class RandomUniform(
    private val maxVal: Float = 1.0f,
    private val minVal: Float = 0.0f,
    private val seed: Long = 12L
) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        require(minVal <= maxVal) { "The minVal parameter value must be less or equal than maxVal parameter value." }

        val seeds = longArrayOf(seed, 0L)
        var distOp: Operand<Float> = tf.random.statelessRandomUniform(shape, tf.constant(seeds), getDType())
        if (minVal == 0.0f) {
            if (minVal != 1.0f) {
                distOp = tf.math.mul(distOp, tf.constant(maxVal))
            }
        } else {
            distOp = tf.math.mul(distOp, tf.dtypes.cast(tf.constant(maxVal - minVal), getDType()))
            distOp = tf.math.add(distOp, tf.dtypes.cast(tf.constant(minVal), getDType()))
        }

        return tf.withName(name).identity(distOp)
    }

    override fun toString(): String {
        return "RandomUniform(maxVal=$maxVal, minVal=$minVal, seed=$seed)"
    }
}