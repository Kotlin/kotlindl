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
import org.tensorflow.Output
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyAdaMax

private const val FIRST_MOMENT = "m"
private const val SECOND_MOMENT = "v"
private val FIRST_BETA_POWER_NAME = defaultOptimizerVariableName("beta1_power")

/**
 * Adamax optimizer from Adam paper's Section 7.
 *
 * Updates variable according next formula:
 * ```
 * m_t <- beta1 * m_{t-1} + (1 - beta1) * g
 * v_t <- max(beta2 * v_{t-1}, abs(g))
 * variable <- variable - learning_rate / (1 - beta1^t) * m_t / (v_t + epsilon)
 * ```
 * It is a variant of Adam based on the infinity norm. Default parameters follow those provided in the paper.
 *
 * NOTE: This optimizer works on CPU only. It has known bug on GPU: NaN instead of gradient values https://github.com/tensorflow/tensorflow/issues/26256
 *
 * It is recommended to leave the parameters of this optimizer at their default values.
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [beta1] 0 < beta < 1. Generally close to 1.
 * @property [beta2] 0 < beta < 1. Generally close to 1.
 * @property [epsilon] Float >= 0. Fuzz factor.
 */
public class Adamax(
    private val learningRate: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val epsilon: Float = 1e-07f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {

    private lateinit var epsilonConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var betaOneConst: Constant<Float>
    private lateinit var betaTwoConst: Constant<Float>
    private lateinit var betaOnePower: Variable<Float>

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(beta1 > 0.0f && beta1 < 1.0f) { "Beta1 $beta1 should be in range (0.0; 1.0)." }
        require(beta2 > 0.0f && beta2 < 1.0f) { "Beta2 $beta2 should be in range (0.0; 1.0)." }
        require(epsilon >= 0.0f) { "L2Strength $epsilon should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        betaOneConst = tf.constant(beta1, getDType())
        betaTwoConst = tf.constant(beta2, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        val scope = Scope(graph.tfGraph)

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val firstMomentSlot: Variable<Float> = getSlot(varName, FIRST_MOMENT)
            val secondMomentSlot: Variable<Float> = getSlot(varName, SECOND_MOMENT)

            targets.add(
                ApplyAdaMax.create(
                    scope,
                    variable,
                    firstMomentSlot,
                    secondMomentSlot,
                    betaOnePower,
                    learningRateConst,
                    betaOneConst,
                    betaTwoConst,
                    epsilonConstant,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    ApplyAdaMax.useLocking(true)
                )
            )
        }

        val betaOnePowerInit = tf
            .assign(betaOnePower, tf.math.mul(betaOnePower, betaOneConst))

        graph.addOptimizerVariableInitializer(betaOnePowerInit)
        graph.addOptimizerVariable(betaOnePower)

        return targets
    }

    private fun createAdamaxSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val firstMomentInitializerName = defaultInitializerOpName(createName(v, FIRST_MOMENT))
        val firstMomentInitializer =
            tf.withName(firstMomentInitializerName).fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), FIRST_MOMENT, firstMomentInitializer)

        val secondMomentInitializerName = defaultInitializerOpName(createName(v, SECOND_MOMENT))
        val secondMomentInitializer = tf.withName(secondMomentInitializerName)
            .fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), SECOND_MOMENT, secondMomentInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdamaxSlot(graph, tf, v.asOutput())
        }
        betaOnePower = tf.withName(FIRST_BETA_POWER_NAME).variable(Shape.scalar(), getDType())
        val betaOnePowerAssignName = defaultAssignOpName(FIRST_BETA_POWER_NAME)

        val betaOnePowerInit: Assign<*> = tf.withName(betaOnePowerAssignName)
            .assign(
                betaOnePower,
                tf.withName(defaultInitializerOpName(FIRST_BETA_POWER_NAME)).constant(beta1, getDType())
            )
        graph.addOptimizerVariableInitializer(betaOnePowerInit)
    }

    override val optimizerName: String get() = "Adamax"

    override val isRunningOnGPU: Boolean get() = false
}
