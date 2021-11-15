package org.jetbrains.kotlinx.dl.api.core.initializer

import org.jetbrains.kotlinx.dl.api.core.exception.IdentityDimensionalityException
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Initializer that generates the identity matrix.
 * This initializer is only usable for generating 2D matrices.
 * Although identity matrices are undefined for non-square matrices an "identity" matrix is generated.
 * E.g. 2 x 3 "identity" matrix
 *   ==> [[ 1.,  0.,  0.],
 *        [ 0.,  1.,  0.]]
 *
 * @property [gain] Identity matrix is multiply by this factor
 * @constructor Creates a [Identity] initializer.
 * @throws [IdentityDimensionalityException] if is not 2D matrix created.
 */
public class Identity(
    public val gain: Float = 1.0f
) : Initializer() {
    override fun initialize(fanIn: Int, fanOut: Int, tf: Ops, shape: Operand<Int>, name: String): Operand<Float> {
        val dimensions = shape.asOutput().shape().size(0)
        if (dimensions != 2L) throw IdentityDimensionalityException(dimensions)

        val minSize = tf.reduceMin(shape, tf.constant(0))
        val reshapedMinSize = tf.reshape(minSize, tf.constant(intArrayOf(1)))
        val diag = tf.tile(tf.constant(floatArrayOf(gain)), reshapedMinSize)

        val zeros = tf.withName(name).zeros(shape, getDType())
        return tf.matrixSetDiagV2(zeros, diag, tf.constant(0))
    }

    override fun toString(): String {
        return "Identity(scale=$gain)"
    }
}
