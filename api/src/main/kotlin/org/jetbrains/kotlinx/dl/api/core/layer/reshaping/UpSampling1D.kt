/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Squeeze

/**
 * Upsampling layer for 1D input.
 *
 * Repeats each step of the second axis [size] times.
 *
 * @property [size] Upsampling factor (i.e. number of repeats).
 */
public class UpSampling1D(
    public val size: Int = 2,
    name: String = "",
) : AbstractUpSampling(
    sizeInternal = intArrayOf(size),
    interpolationInternal = InterpolationMethod.NEAREST,
    name = name,
) {
    init {
        require(size > 0) {
            "The upsampling size should be a positive integer."
        }
    }

    protected override fun computeUpSampledShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) * size,
            inputShape.size(2)
        )
    }

    protected override fun upSample(tf: Ops, input: Operand<Float>): Operand<Float> {
        val inputShape = input.asOutput().shape()
        // First convert the input tensor to a 4D tensor by adding a new axis at the end, so
        // the low-level resize op could be used directly. After resizing, the added
        // axis will be removed.
        val newSize = tf.constant(
            intArrayOf(
                inputShape.size(1).toInt() * size, inputShape.size(2).toInt()
            )
        )
        val expandedInput = tf.expandDims(input, tf.constant(3))
        val resized = tf.image.resizeNearestNeighbor(expandedInput, newSize)
        return tf.squeeze(resized, Squeeze.axis(listOf(3L)))
    }

    override fun toString(): String =
        "UpSampling1D(size=$size)"
}