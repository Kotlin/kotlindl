/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import org.tensorflow.Tensor
import java.nio.FloatBuffer
import java.nio.LongBuffer

/**
 * Provides utility methods for converting tensors in the [TensorResult] to the common data types.
 */
public object TensorFlowInferenceResultConverter : InferenceResultConverter<TensorResult> {
    override fun getFloatArray(result: TensorResult, index: Int): FloatArray {
        return result.getFloatArray(index)
    }

    override fun getLongArray(result: TensorResult, index: Int): LongArray {
        return result.getLongArray(index)
    }
}

/**
 * Returns the output at [index] as a [FloatArray].
 */
public fun TensorResult.getFloatArray(index: Int): FloatArray = tensors[index].toFloatArray()

/**
 * Returns the output at [index] as a [LongArray].
 */
public fun TensorResult.getLongArray(index: Int): LongArray = tensors[index].toLongArray()

/** Copies tensor data to float array. */
public fun Tensor<*>.toFloatArray(): FloatArray {
    val buffer = FloatBuffer.allocate(numElements())
    writeTo(buffer)
    return buffer.array()
}

/** Copies tensor data to long array. */
public fun Tensor<*>.toLongArray(): LongArray {
    val buffer = LongBuffer.allocate(numElements())
    writeTo(buffer)
    return buffer.array()
}

/** Copies tensor to multidimensional float array. Array rank is equal to tensor rank. */
public fun Tensor<*>.toMultiDimensionalArray(): Array<*> {
    val shape = this.shape()
    if (shape.isEmpty()) return emptyArray<Any>()
    if (shape.size == 1) return toFloatArray().toTypedArray()
    val dst = when (shape.size) {
        2 -> create2DArray(shape)
        3 -> create3DArray(shape)
        4 -> create4DArray(shape)
        else -> {
            throw UnsupportedOperationException("Parsing for ${shape.size} dimensions is not supported yet.")
        }
    }
    copyTo(dst)
    return dst
}

private fun create2DArray(shape: LongArray) = Array(shape[shape.size - 2].toInt()) { FloatArray(shape.last().toInt()) }
private fun create3DArray(shape: LongArray) = Array(shape[shape.size - 3].toInt()) { create2DArray(shape) }
private fun create4DArray(shape: LongArray) = Array(shape[shape.size - 4].toInt()) { create3DArray(shape) }