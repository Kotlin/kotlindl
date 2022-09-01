/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.shape.toTensorShape
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D3
import org.jetbrains.kotlinx.multik.ndarray.operations.toList

/**
 * Reverse or permute the [axes] of an input tensor.
 *
 * @property [axes] Array of ints, default value is related to the typical transpose task for H, W, C to C, W, H tensor format conversion.
 */
public class Transpose(public var axes: IntArray = intArrayOf(2, 0, 1)) :
    Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    override fun apply(input: Pair<FloatArray, TensorShape>): Pair<FloatArray, TensorShape> {
        val (data, inputShape) = input

        require(inputShape.rank() == axes.size) { "Transpose operation expected input with ${axes.size} dimensions, but got input with ${inputShape.rank()} dimensions" }

        val tensorShape = inputShape.dims().map { it.toInt() }.toIntArray()

        val ndArray = mk.ndarray<Float, D3>(data.toList(), tensorShape)
        val transposed = ndArray.transpose(*axes)

        return transposed.toList().toFloatArray() to transposed.shape.toTensorShape()
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        require(inputShape.rank() == axes.size) { "Transpose operation expected input with ${axes.size} dimensions, but got input with ${inputShape.rank()} dimensions" }

        val dims = axes.map { inputShape.dims()[it] }.toLongArray()
        return TensorShape(dims)
    }
}


/** Image DSL Preprocessing extension.*/
public fun <I> Operation<I, Pair<FloatArray, TensorShape>>.transpose(sharpBlock: Transpose.() -> Unit): Operation<I, Pair<FloatArray, TensorShape>> {
    return PreprocessingPipeline(this, Transpose().apply(sharpBlock))
}
