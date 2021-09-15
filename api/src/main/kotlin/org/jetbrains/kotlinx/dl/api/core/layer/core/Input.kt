/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.core

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.util.DATA_PLACEHOLDER
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder

/**
 * This layer is responsible for the input shape of the built model.
 *
 * First and required layer in [org.jetbrains.kotlinx.dl.api.core.Sequential.of] method.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [Input] layer from [packedDims] representing [input] data shape.
 */
public class Input(vararg dims: Long, name: String = "") : Layer(name) {
    /** Placeholder for input data. */
    public lateinit var input: Placeholder<Float>

    /** Input data dimensions. Rank = 3 or 4 for most popular supported cases. */
    public var packedDims: LongArray = dims

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    /**
     * Extend this function to define placeholder in layer.
     *
     * NOTE: Called instead of [Layer.build].
     *
     * @param [tf] TensorFlow graph API for building operations.
     */
    public fun build(tf: Ops) {
        input = tf.withName(DATA_PLACEHOLDER).placeholder(
            getDType(),
            Placeholder.shape(Shape.make(-1L, *packedDims))
        )
    }

    /**
     * Computes output shape, based on [input] and [Layer] type.
     *
     * NOTE: Called instead of [Layer.computeOutputShape].
     */
    public fun computeOutputShape(): Shape {
        outputShape = TensorShape(input.asOutput().shape())
        return outputShape.toShape()
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return input
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override fun toString(): String {
        return "Input(shape=${packedDims.contentToString()})"
    }
}
