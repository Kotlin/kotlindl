/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Cropping layer for 2D data (e.g. images)
 *
 * Crops input along the second and third dimensions (i.e. spatial dimensions).
 *
 * Input shape: 4D tensor with shape `(batch_size, rows, cols, channels)`.
 *
 * Output shape: 4D tensor with shape `(batch_size, cropped_rows, cropped_cols, channels)`.
 *
 * @property [cropping] An array consisting of two integer arrays of size two which are interpreted as
 * `[[top_crop, bottom_crop], [left_crop, right_crop]]`.
 */
public class Cropping2D(
    public val cropping: Array<IntArray>,
    name: String = "",
) : AbstractCropping(
    croppingInternal = cropping,
    name = name,
) {
    init {
        require(cropping.size == 2) {
            "The cropping should be an array of size 2."
        }
        require(cropping[0].size == 2 && cropping[1].size == 2) {
            "All elements of cropping should be arrays of size 2."
        }
    }

    override fun crop(tf: Ops, input: Operand<Float>): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val cropSize = intArrayOf(
            inputShape.size(1).toInt() - cropping[0][0] - cropping[0][1],
            inputShape.size(2).toInt() - cropping[1][0] - cropping[1][1]
        )
        return tf.slice(
            input,
            tf.constant(intArrayOf(0, cropping[0][0], cropping[1][0], 0)),
            tf.constant(
                intArrayOf(
                    inputShape.size(0).toInt(),
                    cropSize[0],
                    cropSize[1],
                    inputShape.size(3).toInt()
                )
            )
        )
    }

    override fun toString(): String {
        return "Cropping2D(name = $name, cropping=${cropping.contentDeepToString()}, hasActivation = $hasActivation)"
    }
}
