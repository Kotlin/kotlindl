/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.normalization

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.initializer.Ones
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.numElementsInShape
import org.jetbrains.kotlinx.dl.api.core.shape.shapeToLongArray
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * NOTE: This layer is not trainable and does not updates its weights. It's frozen by default.
 *
 * @property [axis] Integer or a list of integers, the axis that should be normalized (typically the features axis).
 * @property [momentum] Momentum for the moving average.
 * @property [center] If True, add offset of beta to normalized tensor. If False, beta is ignored.
 * @property [epsilon] Small float added to variance to avoid dividing by zero.
 * @property [scale]  If True, multiply by gamma. If False, gamma is not used.
 * @property [gammaInitializer]  Initializer for the gamma weight.
 * @property [betaInitializer] Initializer for the beta weight.
 * @property [movingMeanInitializer] Initializer for the moving mean.
 * @property [movingVarianceInitializer] Initializer for the moving variance.
 * @property [name] Custom layer name.
 * @constructor Creates [BatchNorm2] object.
 */
public class BatchNorm2(
    public val axis: List<Int> = arrayListOf(3),
    public val momentum: Double = 0.99,
    public val center: Boolean = true,
    public val epsilon: Double = 0.001,
    public val scale: Boolean = true,
    public val gammaInitializer: Initializer = Ones(),
    public val betaInitializer: Initializer = Zeros(),
    public val movingMeanInitializer: Initializer = Zeros(),
    public val movingVarianceInitializer: Initializer = Ones(),
    name: String = "",
) : Layer(name) {

    init {
        isTrainable = false
    }

    private lateinit var weightShape: Shape
    private var gamma: Variable<Float>? = null
    private var beta: Variable<Float>? = null
    private lateinit var movingMean: Variable<Float>
    private lateinit var movingVariance: Variable<Float>

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        weightShape = Shape.make(inputShape.size(axis[0]))

        if (name.isNotEmpty()) {
            val movingMeanVariableName = batchNormMovingMeanVarName(name)
            val movingVarianceVariableName = batchNormMovingVarianceVarName(name)

            movingMean = tf.withName(movingMeanVariableName).variable(weightShape, getDType())
            movingVariance = tf.withName(movingVarianceVariableName).variable(weightShape, getDType())

            isTrainable = false // TODO: add isTrainable to addWeight method as a flag

            movingMean = addWeight(tf, kGraph, movingMeanVariableName, movingMean, movingMeanInitializer)
            movingVariance =
                addWeight(tf, kGraph, movingVarianceVariableName, movingVariance, movingVarianceInitializer)

            if (scale) {
                val gammaVariableName = batchNormGammaVarName(name)
                gamma = tf.withName(gammaVariableName).variable(weightShape, getDType())
                gamma = addWeight(tf, kGraph, gammaVariableName, gamma!!, gammaInitializer)
            }

            if (center) {
                val betaVariableName = batchNormBetaVarName(name)
                beta = tf.withName(betaVariableName).variable(weightShape, getDType())
                beta = addWeight(tf, kGraph, betaVariableName, beta!!, betaInitializer)
            }
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.withName("BatchNorm")
            .identity(batchNorm(tf, input, gamma, beta, movingMean, movingVariance, tf.constant(epsilon.toFloat())))
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
        return tf.nn.fusedBatchNorm(
            x,
            beta,
            gamma,
            movingMean,
            movingVar
        ).y()
    }

    override val weights: Map<String, Array<*>> get() = extractBatchNormWeights()

    private fun extractBatchNormWeights(): Map<String, Array<*>> {
        val variableNames = mutableListOf<String>()
        variableNames.add(batchNormMovingMeanVarName(name))
        variableNames.add(batchNormMovingVarianceVarName(name))
        if (scale) variableNames.add(batchNormGammaVarName(name))
        if (center) variableNames.add(batchNormBetaVarName(name))
        return extractWeights(variableNames)
    }

    override val hasActivation: Boolean get() = false

    override val paramCount: Int
        get() {
            var paramCount =
                numElementsInShape(shapeToLongArray(weightShape)) + numElementsInShape(shapeToLongArray(weightShape))
            if (scale) paramCount += numElementsInShape(shapeToLongArray(weightShape))
            if (center) paramCount += numElementsInShape(shapeToLongArray(weightShape))
            return paramCount.toInt()
        }

    /** Returns the shape of gamma variable weights. */
    public val gammaShapeArray: LongArray?
        get() {
            return if (scale) TensorShape(weightShape).dims()
            else null
        }

    /** Returns the shape of beta variable weights. */
    public val betaShapeArray: LongArray?
        get() {
            return if (center) TensorShape(weightShape).dims()
            else null
        }

    /** Returns the shape of movingMean variable weights. */
    public val movingMeanShapeArray: LongArray
        get() = TensorShape(weightShape).dims()

    /** Returns the shape of movingVariance variable weights. */
    public val movingVarianceShapeArray: LongArray
        get() = TensorShape(weightShape).dims()

    override fun toString(): String {
        return "BatchNorm(axis=$axis, momentum=$momentum, center=$center, epsilon=$epsilon, scale=$scale, gammaInitializer=$gammaInitializer, movingMeanInitializer=$movingMeanInitializer, moving_variance_initializer=$movingVarianceInitializer)"
    }
}


