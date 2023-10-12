/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.defaultAssignOpName
import org.jetbrains.kotlinx.dl.api.core.util.defaultInitializerOpName
import org.jetbrains.kotlinx.dl.api.core.util.defaultOptimizerVariableName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyAdagradDa

private val GLOBAL_STEP = defaultOptimizerVariableName("adagrad-da-global-step")
private const val ACCUMULATOR = "gradient_accumulator"
private const val SQUARED_ACCUMULATOR = "gradient_squared_accumulator"

/**
 * Adagrad Dual Averaging algorithm for sparse linear models.
 *
 * This optimizer takes care of regularization of unseen features in a mini batch
 * by updating them when they are seen with a closed form update rule that is equivalent to having updated them on every mini-batch.
 *
 * AdagradDA is typically used when there is a need for large sparsity in the trained model.
 * This optimizer only guarantees sparsity for linear models.
 * Be careful when using AdagradDA for deep networks
 * as it will require careful initialization of the gradient accumulators for it to train.
 *
 * It is recommended to leave the parameters of this optimizer at their default values.
 *
 * @see <a href="http://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf">
 *     Adaptive Subgradient Methods for Online Learning and Stochastic Optimization:[Duchi et al., 2011]</a>
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [initialAccumulatorValue] Float >= 0. Starting value for the accumulators, must be positive.
 * @property [l1Strength] A float value, must be greater than or equal to zero.
 * @property [l2Strength] A float value, must be greater than or equal to zero.
 */
public class AdaGradDA(
    public val learningRate: Float = 0.1f,
    public val initialAccumulatorValue: Float = 0.01f,
    public val l1Strength: Float = 0.01f,
    public val l2Strength: Float = 0.01f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(initialAccumulatorValue >= 0.0f) { "Initial accumulator value $initialAccumulatorValue should be >= 0.0." }
        require(l1Strength >= 0.0f) { "L1Strength $l1Strength should be >= 0.0." }
        require(l2Strength >= 0.0f) { "L2Strength $l2Strength should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets = mutableListOf<Operand<Float>>()

        val learningRateConst = tf.constant(learningRate, getDType())
        val l1StrengthConst = tf.constant(l1Strength, getDType())
        val l2StrengthConst = tf.constant(l2Strength, getDType())

        val globalStep = tf.withName(GLOBAL_STEP).variable(Shape.scalar(), getDType())
        val globalStepAssignName = defaultAssignOpName(GLOBAL_STEP)
        val globalStepInit: Assign<*> = tf.withName(globalStepAssignName)
            .assign(globalStep, tf.withName(defaultInitializerOpName(GLOBAL_STEP)).constant(0.0f))
        graph.addOptimizerVariableInitializer(globalStepInit)

        for ((i, variable) in weights.withIndex()) {
            val output = variable.asOutput()
            val gradSlot = createSlot(ACCUMULATOR, output, tf, graph)
            val gradSquaredSlot = createSlot(SQUARED_ACCUMULATOR, output, tf, graph)
            targets.add(
                tf.train.applyAdagradDa(
                    variable,
                    gradSlot,
                    gradSquaredSlot,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    learningRateConst,
                    l1StrengthConst,
                    l2StrengthConst,
                    tf.dtypes.cast(globalStep, Long::class.javaObjectType),
                    ApplyAdagradDa.useLocking(true)
                )
            )
        }

        val globalStepInitFinish = tf.assignAdd(globalStep, tf.constant(1.0f))
        graph.addOptimizerVariableAssignAddInitializer(globalStepInitFinish)
        graph.addOptimizerVariable(globalStep)
        return targets
    }

    override val optimizerName: String get() = "AdaGradDA"

    override val isRunningOnGPU: Boolean get() = true
}
