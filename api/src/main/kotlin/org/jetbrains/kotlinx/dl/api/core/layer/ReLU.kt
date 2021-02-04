/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * TODO: add threshold and negative slope fields
 */
public class ReLU(
    public val maxValue: Float,
    name: String = ""
) : Layer(name) {

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.clipByValue(
            input,
            tf.constant(0.0f) as Operand<Float>,
            tf.constant(maxValue)
        ) // TODO: maybe rewrite it via ops with gradients via maximum and etc due to missed grads for clibByValue
    }

    override fun toString(): String {
        return "ReLU(maxValue=$maxValue)"
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0
}
