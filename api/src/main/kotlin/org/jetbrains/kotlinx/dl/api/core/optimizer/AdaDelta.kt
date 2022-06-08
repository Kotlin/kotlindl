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
import org.tensorflow.op.train.ApplyAdadelta

private const val ACCUMULATOR = "accum"
private const val ACCUMULATOR_UPDATE = "accum_update"

/**
 * Adadelta optimizer.
 *
 * Updates variable according next formula:
 * ```
 * accum = rho() * accum + (1 - rho()) * grad.square();
 * update = (update_accum + epsilon).sqrt() * (accum + epsilon()).rsqrt() * grad;
 * update_accum = rho() * update_accum + (1 - rho()) * update.square();
 * var -= update;
 * ```
 *
 * Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving window of gradient updates,
 * instead of accumulating all past gradients.
 * This way, Adadelta continues learning even when many updates have been done.
 * Compared to Adagrad, in the original version of Adadelta you don't have to set an initial learning rate.
 * In this version, initial learning rate and decay factor can be set, as in most other Keras optimizers.
 *
 * It is recommended to leave the parameters of this optimizer at their default values.
 *
 * @see <a href="http://arxiv.org/abs/1212.5701">
 *     Adadelta - an adaptive learning rate method</a>
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [rho] Float >= 0. Adadelta decay factor, corresponding to fraction of gradient to keep at each time step.
 * @property [epsilon] Float >= 0. Fuzz factor.
 */
public class AdaDelta(
    private val learningRate: Float = 0.1f,
    private val rho: Float = 0.95f,
    private val epsilon: Float = 1e-8f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var epsilonConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var rhoConst: Constant<Float>

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(rho >= 0.0f) { "Rho $rho should be >= 0.0." }
        require(epsilon >= 0.0f) { "Epsilon $epsilon should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()
        rhoConst = tf.constant(rho, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val accumSlot: Variable<Float> = getSlot(varName, ACCUMULATOR)
            val accumUpdateSlot: Variable<Float> = getSlot(varName, ACCUMULATOR_UPDATE)

            targets.add(
                tf.train.applyAdadelta(
                    variable, accumSlot, accumUpdateSlot,
                    learningRateConst,
                    rhoConst,
                    epsilonConstant,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    ApplyAdadelta.useLocking(true)
                )
            )

        }
        return targets
    }

    private fun createAdaDeltaSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))
        val accumulatorInitializer = tf.withName(accumInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumulatorInitializer)

        val accumUpdateInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR_UPDATE))
        val updateInitializer: Operand<Float> = tf.withName(accumUpdateInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR_UPDATE, updateInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaDeltaSlot(graph, tf, v.asOutput())
        }
    }

    override val optimizerName: String get() = "Adadelta"

    override val isRunningOnGPU: Boolean get() = true
}
