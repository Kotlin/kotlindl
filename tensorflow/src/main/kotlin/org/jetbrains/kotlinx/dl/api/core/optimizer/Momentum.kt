/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyMomentum

private const val MOMENTUM = "momentum"

/**
 * Improved version of [SGD] optimizer.
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [momentum] Float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
 * @property [useNesterov] If true, applies Nesterov momentum.
 */
public class Momentum(
    public val learningRate: Float = 0.001f,
    public val momentum: Float = 0.99f,
    public val useNesterov: Boolean = true,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {

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
        val targets = mutableListOf<Operand<Float>>()

        val learningRateConst = tf.constant(learningRate)
        val momentumConst = tf.constant(momentum)

        for ((i, variable) in weights.withIndex()) {
            val slot = createSlot(MOMENTUM, variable.asOutput(), tf, graph)

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

    override val optimizerName: String get() = "Momentum"

    override val isRunningOnGPU: Boolean get() = true
}
