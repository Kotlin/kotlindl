/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyGradientDescent
import java.util.*

/**
 * Stochastic gradient descent optimizer.
 *
 * NOTE: It's not an equivalent for `keras.sgd`, it is a pure SGD with simple 'variable' update by subtracting 'alpha' * 'delta' from it.
 */
public class SGD(
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private var learningRate: Float = 0.2f

    public constructor(learningRate: Float = 0.2f, clipGradient: ClipGradientAction = NoClipGradient()) : this() {
        this.learningRate = learningRate
    }

    init {
        require(learningRate >= 0.0f) { "Learning rate $learningRate should be >= 0.0." }
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        for (i in weights.indices) {
            targets.add(
                tf.train.applyGradientDescent(
                    weights[i],
                    tf.constant(learningRate, getDType()),
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    ApplyGradientDescent.useLocking(true)
                )
            )
        }

        return targets
    }

    override val optimizerName: String get() = "SGD"

    override val isRunningOnGPU: Boolean get() = true
}
