/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant

/**
 * Layer that reshapes inputs into the given shape.
 *
 * Input shape: `(batch_size,) + any shape`.
 *
 * Output shape: `(batch_size,) + target_shape`.
 *
 * @property [targetShape] Target shape. List of integers, does not include the samples dimension (batch size)
 * @property [name] Custom layer name.
 * @constructor Creates [Reshape] object.
 *
 * @since 0.2
 */
public class Reshape(
    public val targetShape: List<Int>,
    name: String = ""
) : Layer(name) {
    private lateinit var units: Constant<Int>

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        units = tf.constant(IntArray(targetShape.size + 1) {
            if (it == 0) -1 else targetShape[it - 1]
        })
    }

    // leaves unknown dimensions unknown
    override fun computeOutputShape(inputShape: Shape): Shape =
        Shape.make(inputShape.size(0), *targetShape.map { it.toLong() }.toLongArray())

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.reshape(input, units)
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "Reshape"
    }
}
