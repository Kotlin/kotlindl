package api.keras.initializers

import api.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

class RandomUniform(
    private val maxVal: Float,
    private val minVal: Float,
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
        require(minVal < maxVal) { "The minVal parameter value must be less than maxVal parameter value." }

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