/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.toTensorShape
import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32

/**
 * Layer that adds a list of inputs.
 *
 * It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape).
 *
 * @property [layerTypeName] Specified layer name used for tf operation alias building.
 */
public abstract class AbstractMerge(public val layerTypeName: String, name: String = "") : Layer(name) {
    override fun build(
        tf: Ops,
        input: Operand<TFloat32>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> {
        throw UnsupportedOperationException("$layerTypeName is not supported in Sequential models.")
    }

    override fun build(
        tf: Ops,
        input: List<Operand<TFloat32>>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> {
        checkInputShapes(input.map { it.asOutput().shape() }) //TODO: crash efficientNet models
        return tf.withName(layerTypeName).identity(mergeFunction(input, tf))
    }

    /** Should be overridden in all AbstractMerge descendants. */
    protected abstract fun mergeFunction(
        input: List<Operand<TFloat32>>,
        tf: Ops
    ): Operand<TFloat32>

    /** Checks shapes of input operands. */
    protected open fun checkInputShapes(inputShapes: List<Shape>) {
        require(inputShapes.size > 1) { "The number of input layers should be more than 1." }
        val firstInputShape = inputShapes.first().toTensorShape()
        for ((index, inputShape) in inputShapes.withIndex()) {
            val currentInputShape = inputShape.toTensorShape()
            require(firstInputShape == currentInputShape) {
                "The shape of first input $firstInputShape should be equal to the shape $currentInputShape at input index $index."
            }
        }
    }

    override val hasActivation: Boolean
        get() = false
}
