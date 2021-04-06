/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Rectified Linear Unit activation function.
 *
 * With default values, it returns element-wise `max(x, 0)`.
 *
 * Otherwise, it follows:
 * ```
 * f(x) = maxValue if x >= maxValue
 * f(x) = x if threshold <= x < maxValue
 * f(x) = negativeSlope * (x - threshold) otherwise
 * ```
 */
public class ReLU(
    /** Maximum activation value. Should be >= 0. */
    public val maxValue: Float? = null,

    /** Negative slope coefficient. Should be >= 0. */
    public val negativeSlope: Float = 0.0f,

    /** Threshold value for threshold activation. */
    public val threshold: Float = 0.0f,

    name: String = ""
) : Layer(name) {

    init {
        require(negativeSlope >= 0.0f) { "Negative slope $negativeSlope should be >= 0.0." }
        require(maxValue == null || maxValue >= 0.0f) { "Max value $maxValue should be >= 0.0." }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {

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
        // alpha is used for leaky relu slope in activations instead of negativeSlope.
        return commonRelu(tf, input = input, alpha = negativeSlope, maxValue = maxValue, threshold = threshold)
    }

    override fun toString(): String {
        return "ReLU(maxValue=$maxValue, negativeSlope=$negativeSlope, threshold=$threshold)"
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0
}
