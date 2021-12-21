/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.defaultInitializerOpName
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyMomentum
import java.util.*

private const val MOMENTUM = "momentum"

/**
 * Improved version of [SGD] optimizer.
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [momentum] Float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
 * @property [useNesterov] If true, applies Nesterov momentum.
 */
public class Momentum(
    private val learningRate: Float = 0.001f,
    private val momentum: Float = 0.99f,
    private val useNesterov: Boolean = true,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var momentumConst: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(momentum >= 0.0f) { "Momentum $momentum should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        learningRateConst = tf.constant(learningRate)
        momentumConst = tf.constant(momentum)

        for (i in weights.indices) {
            val variable = weights[i]

            val slot = getSlot(variable.ref().op().name(), MOMENTUM)

            targets.add(
                tf.train.applyMomentum(
                    variable,
                    slot,
                    learningRateConst,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    momentumConst,
                    ApplyMomentum.useNesterov(useNesterov),
                    ApplyMomentum.useLocking(true)
                )
            )
        }
        return targets
    }

    private fun createMomentumSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val momentumInitializerName = defaultInitializerOpName(createName(v, MOMENTUM))
        val initializer: Operand<Float> = tf.withName(momentumInitializerName)
            .fill(tf.shape(v), tf.constant(0.0f))
        createSlot(graph, tf, v.asOutput(), MOMENTUM, initializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createMomentumSlot(graph, tf, v.asOutput())
        }
    }

    override val optimizerName: String get() = "Momentum"

    override val isRunningOnGPU: Boolean get() = true
}
