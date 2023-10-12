/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyCenteredRmsProp
import org.tensorflow.op.train.ApplyRmsProp

private const val RMS = "rms"
private const val MG = "mg"
private const val MOMENTUM = "momentum"

/**
 * RMSProp optimizer.
 *
 * @property [learningRate] Float >= 0. Learning rate.
 * @property [decay] Float >= 0. Learning rate decay over each update.
 * @property [momentum] Float >= 0. Parameter that accelerates RMSProp in the relevant direction and dampens oscillations.
 * @property [epsilon] Float >= 0. Fuzz factor.
 * @property [centered] Centered or not.
 */
public class RMSProp(
    public val learningRate: Float = 0.001f,
    public val decay: Float = 0.9f,
    public val momentum: Float = 0.0f,
    public val epsilon: Float = 1e-10f,
    public val centered: Boolean = false,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
        require(momentum >= 0.0f) { "Momentum $momentum should be >= 0.0." }
        require(decay >= 0.0f) { "Decay $decay should be >= 0.0." }
        require(epsilon >= 0.0f) { "Epsilon $epsilon should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets = mutableListOf<Operand<Float>>()

        val decayConst = tf.constant(decay, getDType())
        val momentumConst = tf.constant(momentum, getDType())
        val learningRateConst = tf.constant(learningRate, getDType())
        val epsilonConstant = tf.constant(epsilon, getDType())

        for ((i, variable) in weights.withIndex()) {
            val output = variable.asOutput()
            val rmsSlot = createSlot(RMS, output, tf, graph)
            val momentumSlot = createSlot(MOMENTUM, output, tf, graph)

            if (centered) {
                val mgSlot = createSlot(MG, output, tf, graph)
                targets.add(
                    tf.train.applyCenteredRmsProp(
                        variable,
                        mgSlot,
                        rmsSlot,
                        momentumSlot,
                        learningRateConst,
                        decayConst,
                        momentumConst,
                        epsilonConstant,
                        clipGradient.clipGradient(tf, gradients.dy(i)),
                        ApplyCenteredRmsProp.useLocking(true)
                    )
                )
            } else {
                targets.add(
                    tf.train.applyRmsProp(
                        variable,
                        rmsSlot,
                        momentumSlot,
                        learningRateConst,
                        decayConst,
                        momentumConst,
                        epsilonConstant,
                        gradients.dy(i),
                        ApplyRmsProp.useLocking(true)
                    )
                )
            }
        }
        return targets
    }

    override val optimizerName: String get() = "RMSProp"

    override val isRunningOnGPU: Boolean get() = true
}
