/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import org.tensorflow.Result
import org.tensorflow.Tensor
import org.tensorflow.ndarray.FloatNdArray
import org.tensorflow.ndarray.LongNdArray
import org.tensorflow.ndarray.StdArrays

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
 * Materializes the fetched tensors of a [Result] as a list, in fetch order.
 *
 * `Session.Runner.run()` returned a `List<Tensor<?>>` in the TensorFlow 1.15 API; it now returns an
 * [Result] keyed by output name, so index access is used to preserve the original fetch ordering.
 */
public fun Result.toTensorList(): List<Tensor> = (0 until size()).map { get(it) }

private fun Tensor.asFloatNdArray(): FloatNdArray = this as? FloatNdArray
    ?: throw IllegalArgumentException("Expected a float tensor but got ${javaClass.simpleName}.")

/**
 * Returns the scalar value of a float tensor.
 *
 * Replaces `Tensor.floatValue()` from the TensorFlow 1.15 API; typed tensors are now NdArrays.
 */
public fun Tensor.floatValue(): Float = asFloatNdArray().getFloat()

/*
 * The copyTo overloads below replace `Tensor.copyTo(Object)` from the TensorFlow 1.15 API, which
 * dispatched on the destination array by reflection. Each returns the destination so call sites
 * that used the returned value keep working.
 */

/** Copies a rank-1 float tensor into the preallocated [dst] array. */
public fun Tensor.copyTo(dst: FloatArray): FloatArray {
    StdArrays.copyFrom(asFloatNdArray(), dst); return dst
}

/**
 * Copies a rank-2 float tensor into the preallocated [dst] array.
 *
 * Unlike the other ranks this tolerates a [dst] larger than the tensor, because prediction reuses a
 * buffer sized for a full batch across a possibly shorter final batch.
 */
public fun Tensor.copyTo(dst: Array<FloatArray>): Array<FloatArray> {
    val source = StdArrays.array2dCopyOf(asFloatNdArray())
    for (i in 0 until minOf(source.size, dst.size)) {
        source[i].copyInto(dst[i], endIndex = minOf(source[i].size, dst[i].size))
    }
    return dst
}

/** Copies a rank-3 float tensor into the preallocated [dst] array. */
public fun Tensor.copyTo(dst: Array<Array<FloatArray>>): Array<Array<FloatArray>> {
    StdArrays.copyFrom(asFloatNdArray(), dst); return dst
}

/** Copies a rank-4 float tensor into the preallocated [dst] array. */
public fun Tensor.copyTo(dst: Array<Array<Array<FloatArray>>>): Array<Array<Array<FloatArray>>> {
    StdArrays.copyFrom(asFloatNdArray(), dst); return dst
}

/** Copies a rank-5 float tensor into the preallocated [dst] array. */
public fun Tensor.copyTo(dst: Array<Array<Array<Array<FloatArray>>>>): Array<Array<Array<Array<FloatArray>>>> {
    StdArrays.copyFrom(asFloatNdArray(), dst); return dst
}

/**
 * Returns the output at [index] as a [FloatArray].
 */
public fun TensorResult.getFloatArray(index: Int): FloatArray = tensors[index].toFloatArray()

/**
 * Returns the output at [index] as a [LongArray].
 */
public fun TensorResult.getLongArray(index: Int): LongArray = tensors[index].toLongArray()

/**
 * Copies tensor data to float array.
 *
 * A flat copy is taken regardless of the tensor rank, matching the previous behaviour of writing
 * the tensor into a linear [java.nio.FloatBuffer].
 */
public fun Tensor.toFloatArray(): FloatArray {
    val ndArray = this.asFloatNdArray()
    val result = FloatArray(shape().size().toInt())
    var index = 0
    ndArray.scalars().forEach { result[index++] = it.getFloat() }
    return result
}

/** Copies tensor data to long array. */
public fun Tensor.toLongArray(): LongArray {
    val ndArray = this as? LongNdArray
        ?: throw IllegalArgumentException("Expected a long tensor but got ${javaClass.simpleName}.")
    val result = LongArray(shape().size().toInt())
    var index = 0
    ndArray.scalars().forEach { result[index++] = it.getLong() }
    return result
}

/** Copies tensor to multidimensional float array. Array rank is equal to tensor rank. */
public fun Tensor.toMultiDimensionalArray(): Array<*> {
    val ndArray = this.asFloatNdArray()
    return when (val rank = shape().numDimensions()) {
        0 -> emptyArray<Any>()
        1 -> StdArrays.array1dCopyOf(ndArray).toTypedArray()
        2 -> StdArrays.array2dCopyOf(ndArray)
        3 -> StdArrays.array3dCopyOf(ndArray)
        4 -> StdArrays.array4dCopyOf(ndArray)
        else -> throw UnsupportedOperationException("Parsing for $rank dimensions is not supported yet.")
    }
}