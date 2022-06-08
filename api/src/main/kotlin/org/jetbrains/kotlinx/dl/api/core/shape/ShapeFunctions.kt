/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import kotlin.math.abs

/**
 * Creates constant array.
 */
internal fun constArray(tf: Ops, vararg data: Int): Operand<Int> {
    return tf.constant(data)
}

/** Creates shape [Operand] from [Shape]. */
internal fun shapeOperand(tf: Ops, shape: Shape): Operand<Int> {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return tf.constant(shapeArray)
}

/** Extracts dimensions as [IntArray] from [Shape]. */
internal fun Shape.toIntArray(): IntArray {
    val shapeArray = IntArray(numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = size(i).toInt()
    }
    return shapeArray
}

/** Extracts dimensions as [LongArray] from [Shape]. */
internal fun Shape.toLongArray(): LongArray {
    val shapeArray = LongArray(numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = size(i)
    }
    return shapeArray
}

/** Extracts dimensions as [String] from [shape]. */
internal fun shapeArrayToString(shape: Shape): String {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return shapeArray.contentToString()
}

/** Returns first dimension from all dimensions [dims]. */
internal fun head(vararg dims: Long): Long {
    return dims[0]
}

/** Returns last dimensions (except first) from [dims]. */
internal fun tail(vararg dims: Long): LongArray {
    return dims.copyOfRange(1, dims.size)
}

/** Returns last dimensions (except first) from [shape]. */
public fun tail(shape: Shape): LongArray {
    val shapeArray = LongArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i)
    }
    return tail(*shapeArray)
}

/** Creates [Shape] object from a few [Long] values in [dims]. */
internal fun shapeFromDims(vararg dims: Long): Shape {
    return Shape.make(head(*dims), *tail(*dims))
}

/** Returns amount of elements in Tensor with [shape]. */
internal fun numElementsInShape(shape: LongArray): Long {
    var prod = 1L
    for (i in shape.indices) {
        prod *= abs(shape[i])
    }
    return prod
}

/** Returns amount of elements in [Shape]. */
internal fun Shape.numElements(): Long = numElementsInShape(toLongArray())

/**
 * Flattens the given array of float values.
 * @return flattened array
 */
public fun Array<*>.flattenFloats(): FloatArray {
    val result = mutableListOf<Float>()

    fun flatten(array: Any?): Unit = when (array) {
        is FloatArray -> array.forEach { result.add(it) }
        is Array<*> -> array.forEach { flatten(it) }
        else -> throw IllegalArgumentException("Cannot flatten object: '$array'")
    }

    flatten(this)

    return result.toFloatArray()
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elems of any type.
 * If the most inner array does not have any elements its size is missed in result */
private fun getShapeOfArray(data: Array<*>): Shape {
    fun appendPrimitiveArraySize(size: Int, acc: MutableList<Long>): LongArray {
        acc += size.toLong()
        return acc.toLongArray()
    }

    tailrec fun collectDims(data: Array<*>, acc: MutableList<Long>): LongArray {
        val firstElem = data[0] ?: return acc.toLongArray()
        acc += data.size.toLong()
        return when (firstElem) {
            is Array<*> -> collectDims(firstElem, acc)
            is BooleanArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is ByteArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is CharArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is ShortArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is IntArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is LongArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is FloatArray -> appendPrimitiveArraySize(firstElem.size, acc)
            is DoubleArray -> appendPrimitiveArraySize(firstElem.size, acc)
            else -> acc.toLongArray()
        }
    }
    return shapeFromDims(*collectDims(data, mutableListOf()))
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elems of any type.
 * If the most inner array does not have any elements its size is missed in result
 */
internal val Array<*>.shape: Shape get() = getShapeOfArray(this)
