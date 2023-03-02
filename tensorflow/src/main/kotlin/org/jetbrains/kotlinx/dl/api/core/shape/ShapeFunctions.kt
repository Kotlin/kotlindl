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
    return tf.constant(shape.toIntArray())
}

/** Extracts dimensions as [IntArray] from [Shape]. */
internal fun Shape.toIntArray(): IntArray {
    return IntArray(numDimensions()) { size(it).toInt() }
}

/** Extracts dimensions as [LongArray] from [Shape]. */
internal fun Shape.toLongArray(): LongArray {
    return LongArray(numDimensions()) { size(it) }
}

/** Extracts dimensions as a [String] from [Shape]. */
internal fun Shape.contentToString(): String {
    return toLongArray().contentToString()
}

/** Returns first dimension */
public fun Shape.head(): Long {
    return size(0)
}

/** Returns last dimensions (except first). */
public fun Shape.tail(): LongArray {
    return LongArray(numDimensions() - 1) { size(it + 1) }
}

/** Returns amount of elements in [Shape]. */
internal fun Shape.numElements(): Long = numElementsInShape(toLongArray())

/** Creates [Shape] object from a few [Long] values in [dims]. */
internal fun shapeFromDims(vararg dims: Long): Shape {
    return Shape.make(head(*dims), *tail(*dims))
}

/** Converts [TensorShape] to [Shape] object. */
public fun TensorShape.toShape(): Shape {
    val d = dims()
    return Shape.make(head(*d), *tail(*d))
}

/** Converts [Shape] to [TensorShape] object. */
public fun Shape.toTensorShape(): TensorShape {
    return TensorShape(toLongArray())
}

internal fun Shape.copy(): Shape {
    return Shape.make(head(), *tail())
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elems of any type.
 * If the most inner array does not have any elements its size is missed in result */
private fun getShapeOfArray(data: Array<*>): Shape {
    return shapeFromDims(*getDimsOfArray(data))
}

/**
 * @see TensorShape
 */
internal val Array<*>.shape: Shape get() = getShapeOfArray(this)

/** Returns first dimension from all dimensions [dims]. */
internal fun head(vararg dims: Long): Long {
    return dims[0]
}

/** Returns last dimensions (except first) from [dims]. */
internal fun tail(vararg dims: Long): LongArray {
    return dims.copyOfRange(1, dims.size)
}

/** Returns amount of elements in Tensor with [shape]. */
internal fun numElementsInShape(shape: LongArray): Long {
    var prod = 1L
    for (i in shape.indices) {
        prod *= abs(shape[i])
    }
    return prod
}