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
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/** This layer is not trainable and does not updates its weights. */
public class BatchNorm(
    public val axis: List<Int> = arrayListOf(3),
    public val momentum: Double = 0.99,
    public val center: Boolean = true, // TODO: if center false = disable shifting to zero like in batchNorm in TFJS
    public val epsilon: Double = 0.001,
    public val scale: Boolean = true,
    public val gammaInitializer: Initializer = Ones(),
    public val betaInitializer: Initializer = Zeros(),
    public val movingMeanInitializer: Initializer = Zeros(),
    public val movingVarianceInitializer: Initializer = Ones(),
    name: String = "",
) : Layer(name), NoGradients {

    private lateinit var weightShape: Shape
    private var gamma: Variable<Float>? = null
    private lateinit var beta: Variable<Float>
    private lateinit var movingMean: Variable<Float>
    private lateinit var movingVariance: Variable<Float>

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        weightShape = Shape.make(inputShape.size(axis[0]))

        if (name.isNotEmpty()) {
            val betaVariableName = batchNormBetaVarName(name)
            val movingMeanVariableName = batchNormMovingMeanVarName(name)
            val movingVarianceVariableName = batchNormMovingVarianceVarName(name)

            beta = tf.withName(betaVariableName).variable(weightShape, getDType())
            movingMean = tf.withName(movingMeanVariableName).variable(weightShape, getDType())
            movingVariance = tf.withName(movingVarianceVariableName).variable(weightShape, getDType())

            isTrainable = false // TODO: add isTrainable to addWeight method as a flag

            beta = addWeight(tf, kGraph, betaVariableName, beta, betaInitializer)
            movingMean = addWeight(tf, kGraph, movingMeanVariableName, movingMean, movingMeanInitializer)
            movingVariance =
                addWeight(tf, kGraph, movingVarianceVariableName, movingVariance, movingVarianceInitializer)

            if (scale) {
                val gammaVariableName = batchNormGammaVarName(name)
                gamma = tf.withName(gammaVariableName).variable(weightShape, getDType())
                gamma = addWeight(tf, kGraph, gammaVariableName, gamma!!, gammaInitializer)
            }
        }
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    /**
     * def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
     * # Compute reciprocal of square root of the moving variance element-wise
     * inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
     * # Scale and shift
     * inv *= gamma
     * Y = X * inv + (beta - moving_mean * inv)
     * return Y
     */
    // TODO: https://github.com/SciSharp/TensorFlow.NET/blob/v0.30-keras/src/TensorFlowNET.Keras/Layers/Normalization/BatchNormalization.cs
    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.withName("BatchNorm")
            .identity(batchNorm(tf, input, gamma, beta, movingMean, movingVariance, tf.constant(epsilon.toFloat())))
    }

    private fun batchNorm(
        tf: Ops,
        x: Operand<Float>,
        gamma: Variable<Float>?,
        beta: Operand<Float>,
        movingMean: Operand<Float>,
        movingVar: Operand<Float>,
        eps: Operand<Float>,
    ): Operand<Float> {
        var inv: Operand<Float> = tf.math.rsqrt(tf.math.add(movingVar, eps))

        if (scale) inv = tf.math.mul(inv, gamma!!)

        return tf.math.add(
            tf.math.mul(x, inv),
            tf.math.sub(beta, tf.math.mul(movingMean, inv))
        )
    }

    // TODO: return real weights
    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 4

    override fun toString(): String {
        return "BatchNorm(axis=$axis, momentum=$momentum, center=$center, epsilon=$epsilon, scale=$scale, gammaInitializer=$gammaInitializer, movingMeanInitializer=$movingMeanInitializer, moving_variance_initializer=$movingVarianceInitializer)"
    }

    public val weightShapeArray: LongArray get() = TensorShape(weightShape).dims()
}
