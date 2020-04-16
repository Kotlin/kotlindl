package tf_api.keras

import org.tensorflow.Shape

class TensorShape() {
    private lateinit var dims: LongArray

    constructor(shape: Shape) : this() {
        dims = dimsFromShape(shape)
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
    fun isKnown(i: Int): Boolean {
        return !dims[i].equals(-1)
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

    private fun head(vararg dims: Long): Long {
        return dims[0]
    }

    private fun tail(vararg dims: Long): LongArray {
        return dims.copyOfRange(1, dims.size)
    }

    fun toShape(): Shape {
        return Shape.make(head(*dims), *tail(*dims))
    }
}