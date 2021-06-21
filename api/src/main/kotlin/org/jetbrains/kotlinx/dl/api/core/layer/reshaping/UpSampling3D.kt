/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class UpSampling3D(
    public val size: IntArray = intArrayOf(2, 2, 2),
    name: String = "",
) : AbstractUpSampling(
    sizeInternal = size,
    interpolationInternal = InterpolationMethod.NEAREST,
    name = name,
) {
    init {
        require(size.size == 3) {
            "The upsampling size should be an array of three elements."
        }

        require(size.all { it > 0 }) {
            "All the upsampling size factors should be positive integers."
        }
    }

    protected override fun computeUpSampledShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) * size[0],
            inputShape.size(2) * size[1],
            inputShape.size(3) * size[2],
            inputShape.size(4)
        )
    }

    protected override fun upSample(tf: Ops, input: Operand<Float>): Operand<Float> {
        var upSampled = repeat(tf, input, repeats = size[0], axis = 1)
        upSampled = repeat(tf, upSampled, repeats = size[1], axis = 2)
        upSampled = repeat(tf, upSampled, repeats = size[2], axis = 3)
        return upSampled
    }

    override fun toString(): String =
        "UpSampling3D(size=$size)"
}