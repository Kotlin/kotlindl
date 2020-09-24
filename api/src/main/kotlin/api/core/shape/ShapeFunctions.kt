package api.core.shape

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import kotlin.math.abs

/**
 * Creates constant array.
 */
fun constArray(tf: Ops, vararg data: Int): Operand<Int> {
    return tf.constant(data)
}

/** Creates shape [Operand] from [Shape]. */
fun shapeOperand(tf: Ops, shape: Shape): Operand<Int> {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return tf.constant(shapeArray)
}

/** Extracts dimensions as [IntArray] from [Shape]. */
fun shapeToIntArray(shape: Shape): IntArray {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return shapeArray
}

/** Extracts dimensions as [LongArray] from [Shape]. */
fun shapeToLongArray(shape: Shape): LongArray {
    val shapeArray = LongArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i)
    }
    return shapeArray
}

/** Extracts dimensions as [String] from [shape]. */
fun shapeArrayToString(shape: Shape): String {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return shapeArray.contentToString()
}

/** Returns first dimension from all dimensions [dims]. */
fun head(vararg dims: Long): Long {
    return dims[0]
}

/** Returns last dimensions (except first) from [dims]. */
fun tail(vararg dims: Long): LongArray {
    return dims.copyOfRange(1, dims.size)
}

/** Returns last dimensions (except first) from [shape]. */
fun tail(shape: Shape): LongArray {
    val shapeArray = LongArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i)
    }
    return tail(*shapeArray)
}

/** Creates [Shape] object from a few [Long] values in [dims]. */
fun shapeFromDims(vararg dims: Long): Shape {
    return Shape.make(head(*dims), *tail(*dims))
}

/** Returns amount of elements in Tensor with [shape]. */
fun numElementsInShape(shape: LongArray): Long {
    var prod = 1L
    for (i in shape.indices) {
        prod *= abs(shape[i])
    }
    return prod
}

/** Reshapes 2D array of floats to 1D array of floats. */
fun reshape2DTo1D(dst: Array<FloatArray>, size: Int): FloatArray {
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
fun reshape3DTo1D(dst: Array<Array<FloatArray>>, size: Int): FloatArray {
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
fun reshape4DTo1D(dst: Array<Array<Array<FloatArray>>>, size: Int): FloatArray {
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