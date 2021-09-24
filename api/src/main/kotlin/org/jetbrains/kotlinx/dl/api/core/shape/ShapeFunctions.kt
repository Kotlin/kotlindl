/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
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

/** Reshapes 2D array of floats to 1D array of floats. */
public fun reshape2DTo1D(dst: Array<FloatArray>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0

    for (i in dst.indices) {
        for (j in dst[i].indices) {
            result[pos] = dst[i][j]
            pos++
        }
    }

    return result
}

/** Reshapes 3D array of floats to 1D array of floats. */
public fun reshape3DTo1D(dst: Array<Array<FloatArray>>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0
    for (i in dst.indices) {
        for (j in dst[i].indices) {
            for (k in dst[i][j].indices) {
                result[pos] = dst[i][j][k]
                pos++
            }

        }
    }
    return result
}

/** Reshapes 4D array of floats to 1D array of floats. */
public fun reshape4DTo1D(dst: Array<Array<Array<FloatArray>>>, size: Int): FloatArray {
    val result = FloatArray(size) { 0.0f }

    var pos = 0
    for (i in dst.indices) {
        for (j in dst[i].indices) {
            for (k in dst[i][j].indices) {
                for (m in dst[i][j][k].indices) {
                    result[pos] = dst[i][j][k][m]
                    pos++
                }
            }
        }
    }
    return result
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elems of any type.
 * If the most inner array does not have any elements its size is missed in result */
internal fun getShapeOfArray(data: Array<*>): Shape {
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

/** Shape property of standard JVM array for better readability of code */
internal val Array<*>.shape: Shape get() = getShapeOfArray(this)

/**
 * Create an array of arrays (of arrays...) of Floats with specified [shape] and
 * initialized with given [initValue]. When the number of dimensions in result tensor
 * is bigger than 1, the last dimension array is FloatArray (instead of Array<Float>).
 */
internal fun getFloatArrayOfShape(shape: Shape, initValue: Float = 0.0f): Array<*> {
    fun getFloatArrayOfShape(shape: Shape, dimIndex: Int): Any = if (shape.numDimensions() - 1 == dimIndex) {
        FloatArray(shape.size(dimIndex).toInt()) { initValue }
    } else {
        Array(shape.size(dimIndex).toInt()) { getFloatArrayOfShape(shape, dimIndex + 1) }
    }
    return if (shape.numDimensions() == 1) {
        Array(shape.size(0).toInt()) { initValue }
    } else {
        getFloatArrayOfShape(shape, 0) as Array<*>
    }
}

internal fun Any?.castArray(): Array<*> = this as Array<*>

/** Cast Array<*> to Array<T> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast2D(): Array<T> =
    this.map { it as T }.toTypedArray()

/** Cast Array<*> to Array<Array<T>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast3D(): Array<Array<T>> =
    this.map { it.castArray().cast2D<T>() }.toTypedArray()

/** Cast Array<*> to Array<Array<Array<T>>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast4D(): Array<Array<Array<T>>> =
    this.map { it.castArray().cast3D<T>() }.toTypedArray()

/** Cast Array<*> to Array<Array<Array<Array<T>>>> when sure about its dimensions where usually T is [FloatArray] */
internal inline fun <reified T> Array<*>.cast5D(): Array<Array<Array<Array<T>>>> =
    this.map { it.castArray().cast4D<T>() }.toTypedArray()
