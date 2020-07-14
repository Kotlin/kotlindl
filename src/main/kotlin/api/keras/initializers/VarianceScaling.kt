package api.keras.initializers

import api.keras.shape.shapeToLongArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import kotlin.math.max
import kotlin.math.sqrt

class GlorotNormal<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.TRUNCATED_NORMAL, seed = seed)

class GlorotUniform<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.UNIFORM, seed = seed)

class HeNormal<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed)

class HeUniform<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed)

class LeCunNormal<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed)

class LeCunUniform<T : Number>(
    private val seed: Long
) : VarianceScaling<T>(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed)

open class VarianceScaling<T : Number>(
    private val scale: Double = 1.0,
    private val mode: Mode = Mode.FAN_IN,
    private val distribution: Distribution = Distribution.TRUNCATED_NORMAL,
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
        require(scale > 0.0) { "The 'scale' parameter value must be more than 0.0." }
        var lscale = scale

        // TODO: need to decide - extract from shape or pass as parameters from each layer (where it should be defined)
        //val (fanIn, fanOut) = computeInOut(shape.asOutput().shape())
        //println(" $fanIn $fanOut")

        val fanIn = funIn.toDouble()
        val fanOut = funOut.toDouble()

        lscale /= when (mode) {
            Mode.FAN_IN -> max(1.0, fanIn)
            Mode.FAN_OUT -> max(1.0, fanOut)
            Mode.FAN_AVG -> max(1.0, (fanIn + fanOut) / 2.0)
        }

        val distOp: Operand<T>
        val mulOp: Operand<T>
        val stddev: Double
        val seeds = longArrayOf(seed, 0L)
        when (distribution) {
            Distribution.TRUNCATED_NORMAL -> {
                distOp = tf.random.statelessTruncatedNormal(shape, tf.constant(seeds), dtype)
                stddev = sqrt(lscale) / .87962566103423978
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype))
            }
            Distribution.UNTRANCATED_NORMAL -> {
                distOp = tf.random.statelessRandomNormal(shape, tf.constant(seeds), dtype)
                stddev = sqrt(lscale)
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype))
            }
            Distribution.UNIFORM -> {
                distOp = tf.random.statelessRandomUniform(shape, tf.constant(seeds), dtype)
                stddev = sqrt(3.0 * lscale)
                mulOp = tf.withName(name).math.mul(distOp, tf.dtypes.cast(tf.constant(stddev), dtype))
            }
        }
        return mulOp

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
    TRUNCATED_NORMAL, UNTRANCATED_NORMAL, UNIFORM
}