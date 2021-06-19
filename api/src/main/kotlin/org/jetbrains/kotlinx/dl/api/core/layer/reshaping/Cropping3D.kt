/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Cropping layer for 3D data (e.g. video, spatio-temporal)
 *
 * Crops input along the second, third and forth dimensions.
 *
 * @property [cropping] An array consisting of three integer arrays of size two where each array indicates
 * the number of elements to remove from the beginning and end of the corresponding cropping axis.
 */
public class Cropping3D(
    public val cropping: Array<IntArray>,
    name: String = ""
) : AbstractCropping(
    croppingInternal = cropping,
    name = name
) {
    init {
        require(cropping.size == 3) {
            "The cropping should be an array of size 3."
        }
        require(cropping.all { it.size == 2 }) {
            "All elements of cropping should be arrays of size 2."
        }
    }

    override fun computeCroppedShape(inputShape: Shape): Shape {
        return Shape.make(
            inputShape.size(0),
            inputShape.size(1) - cropping[0][0] - cropping[0][1],
            inputShape.size(2) - cropping[1][0] - cropping[1][1],
            inputShape.size(3) - cropping[2][0] - cropping[2][1],
            inputShape.size(4)
        )
    }

    override fun crop(tf: Ops, input: Operand<Float>): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val cropSize = intArrayOf(
            inputShape.size(1).toInt() - cropping[0][0] - cropping[0][1],
            inputShape.size(2).toInt() - cropping[1][0] - cropping[1][1],
            inputShape.size(3).toInt() - cropping[2][0] - cropping[2][1]
        )
        return tf.slice(
            input,
            tf.constant(intArrayOf(0, cropping[0][0], cropping[1][0], cropping[2][0], 0)),
            tf.constant(
                intArrayOf(
                    inputShape.size(0).toInt(),
                    cropSize[0],
                    cropSize[1],
                    cropSize[2],
                    inputShape.size(4).toInt()
                )
            )
        )
    }

    override fun toString(): String =
        "Cropping3D(cropping=${cropping.contentToString()})"
}