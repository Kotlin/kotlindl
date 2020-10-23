/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package api.core.shape

import org.tensorflow.Shape
import kotlin.math.abs

/**
 * Helper wrapper of [Shape] class with helper methods.
 *
 * NOTE: Developer API.
 * TODO: Create extension functions for [Shape] object.
 */
internal class TensorShape() {
    private lateinit var dims: LongArray

    constructor(shape: Shape) : this() {
        dims = dimsFromShape(shape)
    }

    /**
     * Creates a new `TensorShape` with the given dimensions.
     *
     * @param dims The sizes of the remaining dimensions
     */
    constructor(dims: LongArray) : this() {
        this.dims = dims
    }

    /**
     * Creates a new `TensorShape` with the given dimensions.
     *
     * @param firstDimension The size of the first dimension
     * @param dims The sizes of the remaining dimensions
     */
    constructor(firstDimension: Long, vararg dims: Long) : this() {
        this.dims = LongArray(dims.size + 1)
        this.dims[0] = firstDimension
        System.arraycopy(dims, 0, this.dims, 1, dims.size)
    }

    private fun numDimensions(): Int {
        return dims.size
    }

    /** Returns amount of elements in Tensor with the given shape. */
    fun numElements(): Long {
        var prod = 1L
        for (i in 0 until numDimensions()) {
            prod *= abs(dims[i])
        }
        return prod
    }

    /** Returns the rank of this shape.  */
    fun rank(): Int {
        return dims.size
    }

    /** Returns the array of dimensions representing this shape.  */
    fun dims(): LongArray {
        return dims
    }

    /**
     * Returns the value of a dimension
     *
     * @param i The index at which to retrieve a dimension.
     * @return The size of dimension i
     */
    operator fun get(i: Int): Long {
        return dims[i]
    }

    /**
     * Test whether dimension i in this shape is known
     *
     * @param i Target dimension to test
     * @return Whether dimension i is unknown (equal to -1)
     */
    private fun isKnown(i: Int): Boolean {
        return dims[i] != -1L
    }

    /**
     * Throw an exception if dimension i is unknown.
     *
     * @param i Target dimension to test
     * @throws IllegalStateException if dimension i is unknown
     */
    fun assertKnown(i: Int) {
        check(isKnown(i)) { "Dimension $i in shape needs to be known." }
    }

    /**
     * Replace dimension i with a new dimension size.
     *
     * @param i The target dimension to change.
     * @param dim The new dimension size.
     * @return The new changed TensorShape
     */
    fun replace(i: Int, dim: Long): TensorShape {
        dims[i] = dim
        return this
    }

    /**
     * Replace the last dimension with a new dimension size.
     *
     * @param dim New size for the last dimensions
     * @return The new changed TensorShape
     */
    fun replaceLast(dim: Long): TensorShape {
        return replace(dims.size - 1, dim)
    }

    /**
     * Replace the first dimension with a new dimension size.
     *
     * @param dim New size for first dimension
     * @return The new changed TensorShape.
     */
    fun replaceFirst(dim: Long): TensorShape {
        return replace(0, dim)
    }

    /**
     * Get the size of a target dimension.
     *
     * @param i Target dimension.
     * @return The size of dimension i
     */
    fun size(i: Int): Long {
        return dims[i]
    }

    /**
     * Augment this TensorShape by appending more dimensions to it.
     *
     * @param dims The new dimensions to incorporate
     * @return The new changed TensorShape
     */
    fun concatenate(vararg dims: Long): TensorShape {
        this.dims = concatenate(this.dims, *dims)
        return this
    }

    private fun dimsFromShape(shape: Shape): LongArray {
        val dims = LongArray(shape.numDimensions())
        for (i in 0 until shape.numDimensions()) {
            dims[i] = shape.size(i)
        }
        return dims
    }

    private fun concatenate(first: LongArray, vararg last: Long): LongArray {
        val dims = LongArray(first.size + last.size)
        System.arraycopy(first, 0, dims, 0, first.size)
        System.arraycopy(last, 0, dims, first.size, last.size)
        return dims
    }

    /** Returns first dimension from all dimensions [dims]. */
    // TODO: to companion
    fun head(vararg dims: Long): Long {
        return dims[0]
    }

    fun head(): Long {
        return dims[0]
    }

    /** Returns last dimensions (except first) from [dims]. */
    // TODO: to companion
    fun tail(vararg dims: Long): LongArray {
        return dims.copyOfRange(1, dims.size)
    }

    fun tail(): LongArray {
        return dims.copyOfRange(1, dims.size)
    }

    /** Converts to [Shape] object. */
    fun toShape(): Shape {
        return Shape.make(head(*dims), *tail(*dims))
    }

    override fun toString(): String {
        return dims.contentToString().replace("-1", "None")
    }
}