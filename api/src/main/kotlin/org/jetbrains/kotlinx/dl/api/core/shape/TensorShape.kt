/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import kotlin.math.abs

/**
 * Representation of the tensor shape class with helper methods.
 */
public class TensorShape() {
    private lateinit var dims: LongArray

    /**
     * Creates a new `TensorShape` with the given dimensions.
     *
     * @param dims The sizes of the remaining dimensions
     */
    public constructor(dims: LongArray) : this() {
        this.dims = dims
    }

    /**
     * Creates a new `TensorShape` with the given dimensions.
     *
     * @param firstDimension The size of the first dimension
     * @param dims The sizes of the remaining dimensions
     */
    public constructor(firstDimension: Long, vararg dims: Long) : this() {
        this.dims = LongArray(dims.size + 1)
        this.dims[0] = firstDimension
        dims.copyInto(this.dims, destinationOffset = 1)
    }

    private fun numDimensions(): Int {
        return dims.size
    }

    /** Returns amount of elements in Tensor with the given shape. */
    public fun numElements(): Long {
        var prod = 1L
        for (i in 0 until numDimensions()) {
            prod *= abs(dims[i])
        }
        return prod
    }

    /** Returns the rank of this shape.  */
    public fun rank(): Int {
        return dims.size
    }

    /** Returns the array of dimensions representing this shape.  */
    public fun dims(): LongArray {
        return dims
    }

    /**
     * Returns the value of a dimension
     *
     * @param i The index at which to retrieve a dimension.
     * @return The size of dimension i
     */
    public operator fun get(i: Int): Long {
        return dims[i]
    }

    /**
     * Sets the value of a dimension
     *
     * @param i The index at which to retrieve a dimension.
     */
    public operator fun set(i: Int, value: Long) {
        dims[i] = value
    }

    /**
     * Test whether dimension i in this shape is known
     *
     * @param [i] Target dimension to test
     * @return Whether dimension [i] is unknown (equal to -1)
     */
    private fun isKnown(i: Int): Boolean {
        return dims[i] != -1L
    }

    /**
     * Throw an exception if dimension [i] is unknown.
     *
     * @param [i] Target dimension to test
     * @throws IllegalStateException if dimension [i] is unknown
     */
    public fun assertKnown(i: Int) {
        check(isKnown(i)) { "Dimension $i in shape needs to be known." }
    }

    /**
     * Replace dimension i with a new dimension size.
     *
     * @param i The target dimension to change.
     * @param dim The new dimension size.
     * @return The new changed TensorShape
     */
    public fun replace(i: Int, dim: Long): TensorShape {
        dims[i] = dim
        return this
    }

    /**
     * Replace the last dimension with a new dimension size.
     *
     * @param dim New size for the last dimensions
     * @return The new changed TensorShape
     */
    public fun replaceLast(dim: Long): TensorShape {
        return replace(dims.size - 1, dim)
    }

    /**
     * Replace the first dimension with a new dimension size.
     *
     * @param dim New size for first dimension
     * @return The new changed TensorShape.
     */
    public fun replaceFirst(dim: Long): TensorShape {
        return replace(0, dim)
    }

    /**
     * Get the size of a target dimension.
     *
     * @param i Target dimension.
     * @return The size of dimension i
     */
    public fun size(i: Int): Long {
        return dims[i]
    }

    /**
     * Augment this TensorShape by appending more dimensions to it.
     *
     * @param dims The new dimensions to incorporate
     * @return The new changed TensorShape
     */
    public fun concatenate(vararg dims: Long): TensorShape {
        this.dims = concatenate(this.dims, *dims)
        return this
    }

    private fun concatenate(first: LongArray, vararg last: Long): LongArray {
        val dims = LongArray(first.size + last.size)
        first.copyInto(dims)
        last.copyInto(dims, destinationOffset = first.size)
        return dims
    }

    /** Returns the head dimension. */
    public fun head(): Long {
        return dims[0]
    }

    /** Returns the tail dimension. */
    public fun tail(): LongArray {
        return dims.copyOfRange(1, dims.size)
    }

    override fun toString(): String {
        return dims.contentToString().replace("-1", "None")
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as TensorShape

        return dims.contentEquals(other.dims)
    }

    override fun hashCode(): Int {
        return dims.contentHashCode()
    }

    /** Makes a copy of TensorShape object. */
    public fun clone(): TensorShape {
        return TensorShape(dims)
    }

    /** Check the fact that two shapes has the same values at the same dimensions except one with index [except]. */
    public fun almostEqual(tensorShape: TensorShape, except: Int): Boolean {
        var almostEqual = true
        for (i in 0 until tensorShape.numDimensions()) {
            if (i == except) continue
            if (this[i] != tensorShape[i]) almostEqual = false
        }

        return almostEqual
    }

    public companion object {
        /** Returns first dimension from all dimensions of this array. */
        public fun LongArray.head(): Long = this[0]

        /** Returns last dimensions (except first) from this array. */
        public fun LongArray.tail(): LongArray = copyOfRange(1, size)
    }
}

/**
 * Get shape of array of arrays (of arrays...) of Array of elements of any type.
 * If the most inner array does not have any elements its size is skipped in the result.
 */
public fun getDimsOfArray(data: Array<*>): LongArray {
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
    return collectDims(data, mutableListOf())
}

/**
 * @see getDimsOfArray
 */
public val Array<*>.tensorShape: TensorShape get() = TensorShape(getDimsOfArray(this))

/**
 * Wraps an IntArray to TensorShape.
 */
public fun IntArray.toTensorShape(): TensorShape = TensorShape(this.map(Int::toLong).toLongArray())
