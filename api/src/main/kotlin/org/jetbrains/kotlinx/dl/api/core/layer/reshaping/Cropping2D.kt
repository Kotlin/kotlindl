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

/**
 * Cropping layer for 2D input (e.g. picture).
 * It crops along spatial dimensions, i.e. height and width.
 *
 * Input shape: 4D tensor with shape `(batch_size, rows, cols, channels)`
 *
 * Output shape: 4D tensor with shape: `(batch_size, cropped_rows, cropped_cols, channels)`
 *
 * @property [cropping] tuple of 2 tuples of 2 ints: interpreted as `((top_crop, bottom_crop), (left_crop, right_crop))`.
 * @property [name] Custom layer name.
 * @constructor Creates [Flatten] object.
 *
 * @since 0.2
 */
public class Cropping2D(
    public val cropping: Array<IntArray>,
    name: String = ""
) : Layer(name) {
    private lateinit var shape: LongArray

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        shape = TensorShape(inputShape).dims()

        val rows = inputShape.size(1) - this.cropping[0][0] - this.cropping[0][1]
        val cols = inputShape.size(2) - this.cropping[1][0] - this.cropping[1][1]

        return Shape.make(inputShape.size(0), rows, cols, inputShape.size(3))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val hSliced = sliceAlongAxis(
            tf,
            input,
            this.cropping[0][0],
            shape[1] - this.cropping[0][0] - this.cropping[0][1],
            2
        )
        val sliceAlongAxis = sliceAlongAxis(
            tf,
            hSliced,
            this.cropping[1][0],
            shape[2] - this.cropping[1][1] - this.cropping[1][0],
            3
        )
        return sliceAlongAxis
    }

    // TODO: rewrite more common and multidimensional case https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/backend/tfjs_backend.ts#L226
    private fun sliceAlongAxis(tf: Ops, input: Operand<Float>, start: Int, size: Long, axis: Int): Operand<Float> {
        val inputShape = input.asOutput().shape()

        return when (axis) {
            2 -> tf.slice(
                input, tf.constant(intArrayOf(0, start, 0, 0)), tf.constant(
                    intArrayOf(
                        inputShape.size(0).toInt(),
                        size.toInt(),
                        inputShape.size(2).toInt(),
                        inputShape.size(3).toInt()
                    )
                )
            )
            3 -> tf.slice(
                input, tf.constant(intArrayOf(0, 0, start, 0)), tf.constant(
                    intArrayOf(
                        inputShape.size(0).toInt(),
                        inputShape.size(1).toInt(),
                        size.toInt(),
                        inputShape.size(3).toInt()
                    )
                )
            )
            else -> throw UnsupportedOperationException("$axis is not supported for the slicing in Cropping2D op!")
        }
    }


    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0
}
