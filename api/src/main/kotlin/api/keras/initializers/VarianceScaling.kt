package api.keras.initializers

import api.keras.shape.shapeToLongArray
import api.keras.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import kotlin.math.max
import kotlin.math.sqrt

class GlorotNormal(
    private val seed: Long
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "GlorotNormal(seed=$seed) ${super.toString()}"
    }
}

class GlorotUniform(
    private val seed: Long
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "GlorotUniform(seed=$seed) ${super.toString()}"
    }
}

class HeNormal(
    private val seed: Long
) : VarianceScaling(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "HeNormal(seed=$seed) ${super.toString()}"
    }
}

class HeUniform(
    private val seed: Long
) : VarianceScaling(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "HeUniform(seed=$seed) ${super.toString()}"
    }
}

class LeCunNormal(
    private val seed: Long
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "LeCunNormal(seed=$seed) ${super.toString()}"
    }
}

class LeCunUniform(
    private val seed: Long
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "LeCunUniform(seed=$seed) ${super.toString()}"
    }
}

open class VarianceScaling(
    private val scale: Double = 1.0,
    private val mode: Mode = Mode.FAN_IN,
    private val distribution: Distribution = Distribution.TRUNCATED_NORMAL,
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
        require(scale > 0.0) { "The 'scale' parameter value must be more than 0.0." }
        var lscale = scale

        lscale /= when (mode) {
            Mode.FAN_IN -> max(1.0, fanIn.toDouble())
            Mode.FAN_OUT -> max(1.0, fanOut.toDouble())
            Mode.FAN_AVG -> max(1.0, (fanIn + fanOut).toDouble() / 2.0)
        }

        val distOp: Operand<Float>
        val mulOp: Operand<Float>
        val stddev: Double
        val seeds = longArrayOf(seed, 0L)
        when (distribution) {
            Distribution.TRUNCATED_NORMAL -> {
                distOp = tf.random.statelessTruncatedNormal(shape, tf.constant(seeds), getDType())
                stddev = sqrt(lscale) / .87962566103423978
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), getDType()))
            }
            Distribution.UNTRUNCATED_NORMAL -> {
                distOp = tf.random.statelessRandomNormal(shape, tf.constant(seeds), getDType())
                stddev = sqrt(lscale)
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), getDType()))
            }
            Distribution.UNIFORM -> {
                distOp = tf.random.statelessRandomUniform(shape, tf.constant(seeds), getDType())
                stddev = sqrt(3.0 * lscale)
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), getDType()))
            }
        }
        return mulOp

    }

    override fun toString(): String {
        return "VarianceScaling(scale=$scale, mode=$mode, distribution=$distribution, seed=$seed)"
    }
}

private fun computeInOut(shape: Shape): Pair<Double, Double> {
    val fanIn: Double
    val fanOut: Double

    val dims: LongArray = shapeToLongArray(shape)
    when {
        dims.isEmpty() -> {
            fanOut = 1.0
            fanIn = fanOut
        }
        dims.size == 1 -> {
            fanOut = dims[0].toDouble()
            fanIn = fanOut
        }
        dims.size == 2 -> {
            fanIn = dims[0].toDouble()
            fanOut = dims[1].toDouble()
        }
        else -> {
            var receptiveFieldSize = 1.0
            for (i in dims.size - 2 downTo 0) {
                receptiveFieldSize *= dims[i]
            }
            fanIn = dims[dims.size - 2] * receptiveFieldSize
            fanOut = dims[dims.size - 1] * receptiveFieldSize
        }
    }
    return Pair(fanIn, fanOut)
}

enum class Mode {
    FAN_IN, FAN_OUT, FAN_AVG
}

enum class Distribution {
    TRUNCATED_NORMAL, UNTRUNCATED_NORMAL, UNIFORM
}