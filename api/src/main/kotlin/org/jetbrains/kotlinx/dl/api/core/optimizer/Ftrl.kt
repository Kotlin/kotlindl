/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.defaultInitializerOpName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyFtrl

private const val ACCUMULATOR = "gradient_accumulator"
private const val LINEAR_ACCUMULATOR = "linear_accumulator"

/**
 * Optimizer that implements the FTRL algorithm.
 *
 * Updates variable according next formula:
 * ```
 * m_t <- beta1 * m_{t-1} + (1 - beta1) * g
 * v_t <- max(beta2 * v_{t-1}, abs(g))
 * variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
 * ```
 * This version has support for both online L2 (the L2 penalty given in the paper above)
 * and shrinkage-type L2 (which is the addition of an L2 penalty to the loss function).
 *
 * Check the documentation for the [l2ShrinkageRegularizationStrength]
 * parameter for more details when shrinkage is enabled, in which case gradient is replaced with gradient_with_shrinkage.
 *
 * NOTE: This optimizer works on CPU only. It has known bug on GPU: NaN instead of gradient values https://github.com/tensorflow/tensorflow/issues/26256
 *
 * It is recommended to leave the parameters of this optimizer at their default values.
 *
 * @see <a href="https://research.google.com/pubs/archive/41159.pdf">
 *     See Algorithm 1 of this paper.</a>
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [l1RegularizationStrength] A float value, must be greater than or equal to zero.
 * @property [l2RegularizationStrength] A float value, must be greater than or equal to zero.
 * @property [learningRatePower] A float value, must be less or equal to zero.
 * Controls how the learning rate decreases during training. Use zero for a fixed learning rate.
 * @property [l2ShrinkageRegularizationStrength] A float value, must be greater than or equal to zero.
 * @property [initialAccumulatorValue] The starting value for accumulators. Only zero or positive values are allowed.
 * This differs from L2 above in that the L2 above is a stabilization penalty, whereas this L2 shrinkage is a magnitude penalty.
 * When input is sparse shrinkage will only happen on the active weights.
 */
public class Ftrl(
    private val learningRate: Float = 0.001f,
    private val l1RegularizationStrength: Float = 0.0f,
    private val l2RegularizationStrength: Float = 0.0f,
    private val learningRatePower: Float = -0.5f,
    private val l2ShrinkageRegularizationStrength: Float = 0.0f,
    private var initialAccumulatorValue: Float = 0.0f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    /**  */
    private lateinit var learningRatePowerConst: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var l1RegularizationStrengthConst: Constant<Float>
    private lateinit var l2RegularizationStrengthConst: Constant<Float>
    private lateinit var l2ShrinkageRegularizationStrengthConst: Constant<Float>

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(initialAccumulatorValue >= 0.0f) { "Initial accumulator value $initialAccumulatorValue should be >= 0.0." }
        require(l1RegularizationStrength >= 0.0f) { "L1 Regularization Strength $l1RegularizationStrength should be >= 0.0." }
        require(l2RegularizationStrength >= 0.0f) { "L2 Regularization Strength $l2RegularizationStrength should be >= 0.0." }
        require(learningRatePower <= 0.0f) { "Learning rate power $learningRatePower should be <= 0.0." }
        require(l2ShrinkageRegularizationStrength >= 0.0f) { "L2 Shrinkage Regularization Strength $l2ShrinkageRegularizationStrength should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        l1RegularizationStrengthConst = tf.constant(l1RegularizationStrength, getDType())
        l2RegularizationStrengthConst = tf.constant(l2RegularizationStrength, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        l2ShrinkageRegularizationStrengthConst = tf.constant(l2ShrinkageRegularizationStrength, getDType())
        learningRatePowerConst = tf.constant(learningRatePower, getDType())

        for (i in weights.indices) {

            val variable = weights[i]
            val varName = variable.ref().op().name()

            val accumSlot: Variable<Float> = getSlot(varName, ACCUMULATOR)
            val linearSlot: Variable<Float> = getSlot(varName, LINEAR_ACCUMULATOR)
            val options = ApplyFtrl.useLocking(true)

            targets.add(
                tf.train.applyFtrl(
                    variable,
                    accumSlot,
                    linearSlot,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    learningRateConst,
                    l1RegularizationStrengthConst,
                    l2RegularizationStrengthConst,
                    l2ShrinkageRegularizationStrengthConst,
                    learningRatePowerConst,
                    options
                )
            )
        }

        return targets
    }

    private fun createFtrlSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))
        val accumInitializer = tf.withName(accumInitializerName)
            .fill(tf.shape(v), tf.constant(initialAccumulatorValue))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumInitializer)

        val linearAccumInitializerName = defaultInitializerOpName(createName(v, LINEAR_ACCUMULATOR))
        val linearAccumInitializer = tf.withName(linearAccumInitializerName)
            .fill(tf.shape(v), tf.constant(0.0f))
        createSlot(graph, tf, v.asOutput(), LINEAR_ACCUMULATOR, linearAccumInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createFtrlSlot(graph, tf, v.asOutput())
        }
    }

    override val optimizerName: String get() = "Ftrl"

    override val isRunningOnGPU: Boolean get() = false
}
