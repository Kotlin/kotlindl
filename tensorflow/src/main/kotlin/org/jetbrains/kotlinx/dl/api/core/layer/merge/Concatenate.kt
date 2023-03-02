/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.shape.toTensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Layer that concatenates a list of inputs.
 *
 * It takes as input a list of tensors, all the same shape except
 * for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.
 *
 * @property [axis] Axis along which to concatenate.
 */
public class Concatenate(
    public var axis: Int = 3,
    name: String = ""
) : AbstractMerge("ConcatenateLayer", name), NoGradients {
    override fun checkInputShapes(inputShapes: List<Shape>) {
        require(inputShapes.size > 1) { "The number of input layers should be more than 1." }
        val firstInputShape = inputShapes.first().toTensorShape()
        for ((index, inputShape) in inputShapes.withIndex()) {
            val currentInputShape = inputShape.toTensorShape()
            require(firstInputShape.almostEqual(currentInputShape, except = axis)) {
                "A Concatenate layer requires inputs with matching shapes except for the concat axis $axis. " +
                        "But shapes are the following: shape of first input is $firstInputShape and shape at index $index is $currentInputShape."
            }
        }
    }

    override fun mergeFunction(input: List<Operand<Float>>, tf: Ops): Operand<Float> {
        return tf.concat(input, tf.constant(axis))
    }

    override fun toString(): String {
        return "Concatenate(name = $name, axis=$axis)"
    }
}
