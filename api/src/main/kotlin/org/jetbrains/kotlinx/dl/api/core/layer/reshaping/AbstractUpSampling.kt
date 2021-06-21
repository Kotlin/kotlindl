/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public abstract class AbstractUpSampling(
    public val sizeInternal: IntArray,
    public val interpolationInternal: InterpolationMethod,
    name: String,
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return computeUpSampledShape(inputShape)
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return upSample(tf, input)
    }

    protected abstract fun upSample(tf: Ops, input: Operand<Float>): Operand<Float>

    protected abstract fun computeUpSampledShape(inputShape: Shape): Shape
}

public fun repeat(tf: Ops, value: Operand<Float>, repeats: Int, axis: Int): Operand<Float> {
    val inputShape = value.asOutput().shape()
    val splits = tf.split(tf.constant(axis), value, inputShape.size(axis))
    val multiples = tf.constant(
        IntArray(inputShape.numDimensions()) { if (it == axis) repeats else 1 }
    )
    val repeated = splits.map { tf.tile(it, multiples) }
    // The following check is due to the fact the `tf.concat` raise an error
    // if only one tensor is given as its input.
    if (repeated.size == 1)
        return repeated[0]
    return tf.concat(repeated, tf.constant(axis))
}

public enum class InterpolationMethod(internal val methodName: String) {
    NEAREST("nearest"),
    BILINEAR("bilinear"),
    BICUBIC("bicubic"),
}