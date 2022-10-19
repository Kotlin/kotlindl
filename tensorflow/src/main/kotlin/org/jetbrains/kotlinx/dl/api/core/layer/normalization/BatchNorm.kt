/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.normalization

import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.initializer.Ones
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.*
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.batchNormBetaVarName
import org.jetbrains.kotlinx.dl.api.core.util.batchNormGammaVarName
import org.jetbrains.kotlinx.dl.api.core.util.batchNormMovingMeanVarName
import org.jetbrains.kotlinx.dl.api.core.util.batchNormMovingVarianceVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * NOTE: This layer is not trainable and does not update its weights. It's frozen by default.
 *
 * @property [axis] Integer or a list of integers, the axis that should be normalized (typically the features' axis).
 * @property [momentum] Momentum for the moving average.
 * @property [center] If True, add offset of beta to normalized tensor. If False, beta is ignored.
 * @property [epsilon] Small float added to variance to avoid dividing by zero.
 * @property [scale]  If True, multiply by gamma. If False, gamma is not used.
 * @property [gammaInitializer]  Initializer for the gamma weight.
 * @property [betaInitializer] Initializer for the beta weight.
 * @property [gammaRegularizer] Optional regularizer for the gamma weight.
 * @property [betaRegularizer] Optional regularizer for the beta weight.
 * @property [movingMeanInitializer] Initializer for the moving mean.
 * @property [movingVarianceInitializer] Initializer for the moving variance.
 * @property [name] Custom layer name.
 * @constructor Creates [BatchNorm] object.
 *
 * @since 0.2
 */
public class BatchNorm(
    public val axis: List<Int> = arrayListOf(3),
    public val momentum: Double = 0.99,
    public val center: Boolean = true,
    public val epsilon: Double = 0.001,
    public val scale: Boolean = true,
    public val gammaInitializer: Initializer = Ones(),
    public val betaInitializer: Initializer = Zeros(),
    public val gammaRegularizer: Regularizer? = null,
    public val betaRegularizer: Regularizer? = null,
    public val movingMeanInitializer: Initializer = Zeros(),
    public val movingVarianceInitializer: Initializer = Ones(),
    name: String = "",
) : Layer(name), NoGradients, ParametrizedLayer {
    internal var gamma: KVariable? = null
    internal var beta: KVariable? = null
    internal lateinit var movingMean: KVariable
    internal lateinit var movingVariance: KVariable

    override val variables: List<KVariable>
        get() = listOfNotNull(gamma, beta, movingMean, movingVariance)

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val inputShape = input.asOutput().shape()
        // Compute shapes of kernel and bias matrices
        val weightShape = Shape.make(inputShape.size(axis[0]))

        if (name.isEmpty()) throw RuntimeException("Cannot build BatchNorm layer, because of empty name")

        val fanIn = Int.MIN_VALUE
        val fanOut = Int.MIN_VALUE

        movingMean = createVariable(
            tf,
            batchNormMovingMeanVarName(name),
            weightShape,
            fanIn,
            fanOut,
            movingMeanInitializer,
            null
        )

        movingVariance = createVariable(
            tf,
            batchNormMovingVarianceVarName(name),
            weightShape,
            fanIn,
            fanOut,
            movingVarianceInitializer,
            null
        )

        if (scale) {
            gamma = createVariable(
                tf,
                batchNormGammaVarName(name),
                weightShape,
                fanIn,
                fanOut,
                gammaInitializer,
                gammaRegularizer
            )
        }

        if (center) {
            beta = createVariable(
                tf,
                batchNormBetaVarName(name),
                weightShape,
                fanIn,
                fanOut,
                betaInitializer,
                betaRegularizer
            )
        }

        val tf = tf.withName("BatchNorm")
        return batchNorm(
            tf,
            input,
            gamma?.variable,
            beta?.variable,
            movingMean.variable,
            movingVariance.variable,
            tf.constant(epsilon.toFloat())
        )
    }

    /**
     * ```
     * def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
     *     # Compute reciprocal of square root of the moving variance element-wise
     *     inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
     *     # Scale and shift
     *     inv *= gamma
     *     Y = X * inv + (beta - moving_mean * inv)
     *     return Y
     * ```
     */
    private fun batchNorm(
        tf: Ops,
        x: Operand<Float>,
        gamma: Variable<Float>?,
        beta: Operand<Float>?,
        movingMean: Operand<Float>,
        movingVar: Operand<Float>,
        eps: Operand<Float>,
    ): Operand<Float> {
        var inv: Operand<Float> = tf.math.rsqrt(tf.math.add(movingVar, eps))

        if (scale) inv = tf.math.mul(inv, gamma)

        // NOTE: Y = X * inv + (beta - moving_mean * inv) = X * (inv - moving_mean) + beta
        val xNorm = tf.math.mul(tf.math.sub(x, movingMean), inv)
        return if (center) tf.math.add(xNorm, beta)
        else xNorm
    }

    override fun toString(): String {
        return "BatchNorm(name = $name, isTrainable=$isTrainable, axis=$axis, momentum=$momentum, center=$center, epsilon=$epsilon, scale=$scale, " +
                "gammaInitializer=$gammaInitializer, betaInitializer=$betaInitializer, " +
                "gammaRegularizer=$gammaRegularizer, betaRegularizer=$betaRegularizer, " +
                "movingMeanInitializer=$movingMeanInitializer, movingVarianceInitializer=$movingVarianceInitializer, " +
                "hasActivation=$hasActivation, " +
                "gammaShapeArray=${gamma?.shape}, betaShapeArray=${beta?.shape}, " +
                "movingMeanShapeArray=${movingMean.shape}, movingVarianceShapeArray=${movingVariance.shape})"
    }

    override val hasActivation: Boolean get() = false
}

