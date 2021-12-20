/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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
import org.tensorflow.op.train.ApplyCenteredRmsProp
import org.tensorflow.op.train.ApplyRmsProp
import java.util.*

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
    private val learningRate: Float = 0.001f,
    private val decay: Float = 0.9f,
    private val momentum: Float = 0.0f,
    private val epsilon: Float = 1e-10f,
    private val centered: Boolean = false,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {

    private lateinit var epsilonConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var decayConst: Constant<Float>
    private lateinit var momentumConst: Constant<Float>

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
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        decayConst = tf.constant(decay, getDType())
        momentumConst = tf.constant(momentum, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val rmsSlot: Variable<Float> = getSlot(varName, RMS)
            val momentumSlot: Variable<Float> = getSlot(varName, MOMENTUM)

            if (centered) {
                val mgSlot: Variable<Float> = getSlot(varName, MG)
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

    private fun createRMSPropSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val rmsInitializerName = defaultInitializerOpName(createName(v, RMS))

        val rmsInitializer: Operand<Float> = tf.withName(rmsInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(1.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), RMS, rmsInitializer)

        val momentumInitializerName = defaultInitializerOpName(createName(v, MOMENTUM))
        val momentumInitializer: Operand<Float> = tf.withName(momentumInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), MOMENTUM, momentumInitializer)

        if (centered) {
            val mgInitializerName = defaultInitializerOpName(createName(v, MG))
            val mgInitializer: Operand<Float> = tf.withName(mgInitializerName)
                .fill(
                    tf.shape(v),
                    tf.constant(0.0f)
                )
            createSlot(graph, tf, v.asOutput(), MG, mgInitializer)
        }
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createRMSPropSlot(graph, tf, v.asOutput())
        }
    }

    override val optimizerName: String get() = "RMSProp"

    override val isRunningOnGPU: Boolean get() = true
}
