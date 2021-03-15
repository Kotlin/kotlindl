/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import kotlin.math.abs

/**
 * Layer that reshapes inputs into the given shape.
 *
 * TODO: different calculation for targetShape size 1,2,3
 * @property [targetShape] Target shape. List of integers, does not include the samples dimension (batch size)
 * @property [name] Custom layer name.
 * @constructor Creates [Reshape] object.
 */
public class Reshape(
    public val targetShape: List<Int>,
    name: String = ""
) : Layer(name) {
    private lateinit var units: Constant<Int>

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        val tensorShape = TensorShape(inputShape)
        val amountOfNeuronsInFlattenLayer = (tensorShape.numElements() / abs(tensorShape.size(0))).toInt()
        if (targetShape.size == 3) units = tf.constant(intArrayOf(-1, targetShape[0], targetShape[1], targetShape[2]))
        else if (targetShape.size == 1) units = tf.constant(intArrayOf(-1, targetShape[0]))
        fanIn = tensorShape.numElements().toInt()
        fanOut = amountOfNeuronsInFlattenLayer
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        val tensorShape = TensorShape(inputShape)
        return when (targetShape.size) {
            3 -> Shape.make(
                tensorShape.head(),
                targetShape[0].toLong(),
                targetShape[1].toLong(),
                targetShape[2].toLong()
            )
            2 -> Shape.make(
                tensorShape.head(),
                targetShape[0].toLong(),
                targetShape[1].toLong(),
            )
            1 -> Shape.make(tensorShape.head(), targetShape[0].toLong())
            else -> throw UnsupportedOperationException("Input shape with ${targetShape.size} dimensions is not supported.")
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.reshape(input, units)
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "Reshape"
    }
}
