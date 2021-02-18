/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class Multiply(name: String = "") : Layer(name) {
    public val mergedLayers: List<Layer> = emptyList()

    init {
        inboundLayers = mergedLayers as MutableList<Layer>
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {

    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // TODO: rewrite as in concatenate layer
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return input
    }

    override fun forward(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // TODO: add universal for n inputs (and define merge function) or addN
        // TODO: need to check all output shapes for all inputs (it should be equal)
        return tf.withName("ADD_LAYER").math.mul(input[0], input[1])
    }

    override val weights: List<Array<*>>
        get() = emptyList()
    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
}
