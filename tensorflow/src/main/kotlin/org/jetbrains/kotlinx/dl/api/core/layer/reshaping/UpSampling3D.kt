/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Upsampling layer for 3D input.
 *
 * Repeats the  second, third and forth dimensions of the input by `size[0]`, `size[1]` and
 * `size[2]` times, respectively.
 *
 * Input shape: 5D tensor with shape `(batch_size, dim1, dim2, dim3, depth)`.
 *
 * Output shape: 5D tensor with shape `(batch_size, dim1 * size[0], dim2 * size[1], dim3 * size[2], depth)`.
 *
 * @property [size] Upsampling factor array of size 3 (i.e. number of repeats per each dimension).
 *
 * @since 0.3
 */
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

    protected override fun upSample(tf: Ops, input: Operand<Float>): Operand<Float> {
        var upSampled = input
        repeat(3) {
            if (size[it] > 1)
                upSampled = repeat(tf, upSampled, repeats = size[it], axis = it + 1)
        }
        return upSampled
    }

    override fun toString(): String {
        return "UpSampling3D(name = $name, size=${size.contentToString()}, hasActivation=$hasActivation)"
    }


}
