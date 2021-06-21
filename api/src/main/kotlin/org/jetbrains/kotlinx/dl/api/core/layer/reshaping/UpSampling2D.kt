/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.image.ResizeBilinear

public class UpSampling2D(
    public val size: IntArray = intArrayOf(2, 2),
    public val interpolation: InterpolationMethod = InterpolationMethod.NEAREST,
    name: String = "",
) : AbstractUpSampling(
    sizeInternal = size,
    interpolationInternal = interpolation,
    name = name,
) {
    init {
        require(size.size == 2) {
            "The upsampling size factor should be an array of two elements."
        }

        require(size[0] > 0 && size[1] > 0) {
            "All the upsampling size factors should be positive integers."
        }

        require(interpolation == InterpolationMethod.NEAREST || interpolation == InterpolationMethod.BILINEAR) {
            "The interpolation method should be either of `InterpolationMethod.NEAREST` or `InterpolationMethod.BILINEAR`."
        }
    }

    protected override fun computeUpSampledShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) * size[0],
            inputShape.size(2) * size[1],
            inputShape.size(3)
        )
    }

    protected override fun upSample(tf: Ops, input: Operand<Float>): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val newSize = tf.constant(
            intArrayOf(
                inputShape.size(1).toInt() * size[0],
                inputShape.size(2).toInt() * size[1]
            )
        )
        return when(interpolation) {
            InterpolationMethod.NEAREST -> tf.image.resizeNearestNeighbor(input, newSize)
            InterpolationMethod.BILINEAR ->
                tf.image.resizeBilinear(input, newSize, ResizeBilinear.halfPixelCenters(true))
            else -> throw IllegalArgumentException("The interpolation type interpolation is not supported.")
        }
    }

    override fun toString(): String =
        "UpSampling2D(size=$size, interpolation=$interpolation)"
}
