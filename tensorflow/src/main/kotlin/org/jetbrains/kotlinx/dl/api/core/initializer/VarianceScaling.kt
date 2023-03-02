/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import kotlin.math.max
import kotlin.math.sqrt

/**
 * The Glorot normal initializer, also called Xavier normal initializer.
 *
 * Draw samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / (fan_in + fan_out))`
 * where `fan_in` is the number of input units in
 * the weight tensor and `fan_out` is the number of output units in the weight tensor.
 *
 * @constructor Creates [GlorotNormal] initializer.
 *
 * @see <a href="http://proceedings.mlr.press/v9/glorot10a.html">
 *     Glorot et al., 2010</a>
 */
public class GlorotNormal(
    seed: Long = 12L
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "GlorotNormal(seed=$seed) ${super.toString()}"
    }
}

/**
 * The Glorot uniform initializer, also called Xavier uniform initializer.
 *
 * Draw samples from a uniform distribution within `[-limit, limit]` where `limit`
 * is `sqrt(6 / (fan_in + fan_out))` where `fan_in` is the number of input units
 * in the weight tensor and `fan_out` is the number of output units in the weight tensor.
 *
 * @property [seed] Used to create random seeds.
 * @constructor Creates [GlorotUniform] initializer.
 *
 * @see <a href="http://proceedings.mlr.press/v9/glorot10a.html">
 *     Glorot et al., 2010</a>
 */
public class GlorotUniform(
    seed: Long = 12L
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_AVG, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "GlorotUniform(seed=$seed) ${super.toString()}"
    }
}

/**
 * He normal initializer.
 *
 * Draw samples from a truncated normal distribution centered on 0 with `stddev = sqrt(2 / fan_in)`
 * where `fan_in` is the number of input units in the weight tensor.
 *
 * @property [seed] Used to create random seeds.
 * @constructor Creates [HeNormal] initializer.
 *
 * @see <a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html">
 *     He et al., 2015</a>
 */
public class HeNormal(
    seed: Long = 12L
) : VarianceScaling(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "HeNormal(seed=$seed) ${super.toString()}"
    }
}

/**
 * He uniform variance scaling initializer.
 *
 * Draw samples from a uniform distribution within `[-limit, limit]` where `limit`
 * is `sqrt(6 / fan_in)` where `fan_in` is the number of input units in the weight tensor.
 *
 * @property [seed] Used to create random seeds.
 * @constructor Creates [HeUniform] initializer.
 *
 * @see <a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html">
 *     He et al., 2015</a>
 */
public class HeUniform(
    seed: Long = 12L
) : VarianceScaling(scale = 2.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "HeUniform(seed=$seed) ${super.toString()}"
    }
}

/**
 * LeCun normal initializer.
 *
 * Draw samples from a truncated normal distribution centered on 0 with `stddev = sqrt(1 / fan_in)`
 * where `fan_in` is the number of input units in the weight tensor.
 *
 * @property [seed] Used to create random seeds.
 * @constructor Creates [LeCunNormal] initializer.
 *
 * @see <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">
 *     Self-Normalizing Neural Networks, [Klambauer et al., 2017]</a>
 * @see <a href="https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf">
 *     Efficient Backprop, [Lecun et al., 1998]</a>
 */
public class LeCunNormal(
    seed: Long = 12L
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.TRUNCATED_NORMAL, seed = seed) {
    override fun toString(): String {
        return "LeCunNormal(seed=$seed) ${super.toString()}"
    }
}

/**
 * LeCun uniform initializer.
 *
 * Draw samples from a uniform distribution within [-limit, limit] where `limit` is `sqrt(3 / fan_in)`
 * where `fan_in` is the number of input units in the weight tensor.
 *
 * @property [seed] Used to create random seeds.
 * @constructor Creates [LeCunUniform] initializer.
 *
 * @see <a href="http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf">
 *     Self-Normalizing Neural Networks, [Klambauer et al., 2017]</a>
 * @see <a href="https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf">
 *     Efficient Backprop, [Lecun et al., 1998]</a>
 */
public class LeCunUniform(
    seed: Long = 12L
) : VarianceScaling(scale = 1.0, mode = Mode.FAN_IN, distribution = Distribution.UNIFORM, seed = seed) {
    override fun toString(): String {
        return "LeCunUniform(seed=$seed) ${super.toString()}"
    }
}

/**
 * Initializer capable of adapting its scale to the shape of weights tensors.
 *
 * With `distribution="truncated_normal" or "untruncated_normal"`, samples are
 * drawn from a truncated/untruncated normal distribution with a mean of zero and
 * a standard deviation (after truncation, if used) `stddev = sqrt(scale / n)`
 * where n is:
 * - number of input units in the weight tensor, if mode = "fan_in"
 * - number of output units, if mode = "fan_out"
 * - average of the numbers of input and output units, if mode = "fan_avg"
 *
 * With `distribution="uniform"`, samples are drawn from a uniform distribution
 * within [-limit, limit], with `limit = sqrt(3 * scale / n)`.
 *
 * @property [scale] Scaling factor (should be positive).
 * @property [mode] One of "fan_in", "fan_out", "fan_avg".
 * @property [distribution] Random distribution to use. One of "truncated_normal", "untruncated_normal" and  "uniform".
 * @property [seed] Used to create random seeds.
 * @constructor Creates [VarianceScaling] initializer.
 */
public open class VarianceScaling(
    public val scale: Double = 1.0,
    public val mode: Mode = Mode.FAN_IN,
    public val distribution: Distribution = Distribution.TRUNCATED_NORMAL,
    public val seed: Long = 12L
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

/**
 * Mode.
 */
public enum class Mode {
    /** */
    FAN_IN,

    /** */
    FAN_OUT,

    /** */
    FAN_AVG
}

/**
 * Distribution.
 */
public enum class Distribution {
    /** */
    TRUNCATED_NORMAL,

    /** */
    UNTRUNCATED_NORMAL,

    /** */
    UNIFORM
}
