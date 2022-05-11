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
import org.tensorflow.op.train.ApplyAdagrad

private const val ACCUMULATOR = "accumulator"

/**
 * Adagrad optimizer.
 *
 * Updates variable according next formula:
 * ```
 * accum += grad * grad
 * var -= lr * grad * (1 / sqrt(accum))
 * ```
 *
 * Adagrad is an optimizer with parameter-specific learning rates,
 * which are adapted relative to how frequently a parameter gets
 * updated during training. The more updates a parameter receives, the smaller the updates.
 *
 * It is recommended to leave the parameters of this optimizer at their default values.
 *
 * @see <a href="http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf">
 *     Adaptive Subgradient Methods for Online Learning and Stochastic Optimization</a>
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [initialAccumulatorValue] Decay: Float >= 0. Learning rate decay over each update.
 */
public class AdaGrad(
    private val learningRate: Float = 0.1f,
    private val initialAccumulatorValue: Float = 0.01f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var initialAccumulatorValueConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(initialAccumulatorValue >= 0.0f) { "Rho $initialAccumulatorValue should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        initialAccumulatorValueConstant = tf.constant(initialAccumulatorValue, getDType())
        learningRateConst = tf.constant(learningRate, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val slot: Variable<Float> = getSlot(varName, ACCUMULATOR)

            targets.add(
                tf.train.applyAdagrad(
                    variable,
                    slot,
                    learningRateConst,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    ApplyAdagrad.useLocking(true),
                    ApplyAdagrad.updateSlots(true)
                )
            )
        }
        return targets
    }

    private fun createAdaGradSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))

        val initializer: Operand<Float> = tf.withName(accumInitializerName)
            .fill(tf.shape(v), tf.constant(initialAccumulatorValue))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, initializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaGradSlot(graph, tf, v.asOutput())
        }
    }

    override val optimizerName: String get() = "Adagrad"

    override val isRunningOnGPU: Boolean get() = true
}
