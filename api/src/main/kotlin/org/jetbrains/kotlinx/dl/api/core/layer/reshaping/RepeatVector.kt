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
import org.tensorflow.op.core.Tile

/**
 * Layer that repeats the input [n] times.
 *
 * Input shape: `2D tensor of shape (num_samples, features)`.
 *
 * Output shape: `3D tensor of shape (num_samples, n, features)`.
 *
 * @property n Repetition factor.
 * @property [name] Custom layer name.
 * @constructor Creates [RepeatVector] object.
 *
 * @author Stan van der Bend
 * @since 0.2
 */
public class RepeatVector(
    public val n: Int,
    name: String = ""
) : Layer(name) {

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        require(inputShape.numDimensions() == 2) { "input tensor must have 2 dimensions" }
        val tensorShape = TensorShape(inputShape)
        // TODO: maybe make `n` of type Long?
        return Shape.make(tensorShape[0], n.toLong(), tensorShape[1])
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.repeat(input, n)
    }

    private fun Ops.repeat(input: Operand<Float>, n : Int) : Tile<Float> {
        val x = expandDims(input, constant(1))
        val pattern = stack(listOf(constant(1), constant(n), constant(1)))
        return tile(x, pattern)
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "RepeatVector"
    }
}