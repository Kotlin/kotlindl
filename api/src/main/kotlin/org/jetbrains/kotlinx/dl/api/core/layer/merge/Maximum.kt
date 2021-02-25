/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Layer that computes the maximum (element-wise) a list of inputs.
 *
 * It takes as input a list of tensors, all of the same shape, and returns a single tensor (also of the same shape).
 */
public class Maximum(name: String = "") : Layer(name) {
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
        require(input.size > 1) { "The number of input layers should be more than 1." }

        val firstInputShape = TensorShape(input[0].asOutput().shape())

        for (layer in input) {
            val tensorShape = TensorShape(
                layer.asOutput().shape()
            )
            require(
                firstInputShape == tensorShape
            ) { "The shape of first input $firstInputShape should be equal to the shape $tensorShape of $layer " }
        }

        var output = input[0]
        for (i in 1 until input.size)
            output = tf.math.maximum(output, input[i])

        return tf.withName("MaximumLayer").identity(output)
    }

    override val weights: List<Array<*>>
        get() = emptyList()

    override val hasActivation: Boolean
        get() = false

    override val paramCount: Int
        get() = 0
}
