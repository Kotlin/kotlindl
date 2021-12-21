package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.*
import org.tensorflow.op.linalg.MatMul
import org.tensorflow.op.linalg.Transpose
import org.tensorflow.op.math.Mul
import org.tensorflow.op.math.Rsqrt
import org.tensorflow.op.math.Square

/**
 * Layer that computes a dot product between samples in two tensors.
 *
 * @param axis: Axis along which to take the dot product.
 * @param normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
 */
public class Dot(
    public val axis: IntArray,
    public val normalize: Boolean = false,
    name: String = ""
) : AbstractMerge("Dot", name) {

    public constructor(axis: Int) : this(axis = IntArray(2) { axis })

    override fun mergeFunction(input: List<Operand<Float>>, tf: Ops): Operand<Float> {
        require(input.size == 2) { "A `Dot` layer should be called on exactly 2 input. 'Received: input=${input}" }
        var x1 = input[0]
        var x2 = input[1]
        val axes: IntArray = IntArray(2)
        val scope: Scope = tf.scope()
        for (i in 0..2) {
            if (axis[i] < 0) {
                axes[i] = axis[i] % input[i].asOutput().shape().numDimensions()
            } else {
                axes[i] = axis[i]
            }
        }
        if (normalize) {
            x1 = l2Normalize(scope, x1, intArrayOf(axes[0]))
            x2 = l2Normalize(scope, x2, intArrayOf(axes[1]))
        }
        return batchDot(scope, x1, x2, axes)
    }
}

/**
 * Normalizes a tensor wrt the L2 norm alongside the specified axis.
 *
 * @param scope: Current scope
 * @param x: Operand
 * @param axis: Axis along which to perform normalization
 */
public fun l2Normalize(scope: Scope?, x: Operand<Float>, axis: IntArray?): Operand<Float> {
    val squareSum: Operand<Float> = ReduceSum.create(
        scope,
        Square.create(scope, x),
        Constant.create(scope, axis),
        ReduceSum.keepDims(true)
    )
    val invNorm: Operand<Float> = Rsqrt.create(
        scope,
        org.tensorflow.op.math.Maximum.create(
            scope, squareSum, Constant.create(scope, 1e-12f)
        )
    )
    return Mul.create(scope, x, invNorm)
}

/**
 * Batchwise dot product.
 *
 * @param scope: Current scope
 * @param x: Operand with dimensions>=2
 * @param y: Operand with dimensions>=2
 * @param axis: Axis along which to perform batch dot
 */
public fun batchDot(scope: Scope?, x: Operand<Float>, y: Operand<Float>, axis: IntArray): Operand<Float> {
    val xDim = x.asOutput().shape().numDimensions()
    val yDim = y.asOutput().shape().numDimensions()
    val diff: Int
    var x2: Operand<Float> = x;
    var y2: Operand<Float> = y
    if (xDim > yDim) {
        diff = xDim - yDim
        y2 = Reshape.create(
            scope,
            y,
            Concat.create(
                scope,
                listOf(Shape.create(scope, y)) + List(diff) { Constant.create(scope, 1) },
                Constant.create(scope, 0)
            ),
        )
    } else if (yDim > xDim) {
        diff = yDim - xDim
        x2 = Reshape.create(
            scope,
            x,
            Concat.create(
                scope,
                listOf(Shape.create(scope, x)) + List(diff) { Constant.create(scope, 1) },
                Constant.create(scope, 0)
            ),
        )
    } else {
        diff = 0
    }
    var out: Operand<Float>
    val x2Dim = x2.asOutput().shape().numDimensions()
    val y2Dim = y2.asOutput().shape().numDimensions()
    if (x2Dim == 2 && y2Dim == 2) {
        if (axis[0] == axis[1]) {
            out = ReduceSum.create(scope, Mul.create(scope, x2, y2), Constant.create(scope, axis[0]))
        } else {
            out = ReduceSum.create(
                scope,
                Mul.create(scope, Transpose.create(scope, x2, Constant.create(scope, intArrayOf(1, 0))), y2),
                Constant.create(scope, axis[1])
            )
        }
    } else {
        val adjX = when (axis[0] == x2Dim - 1) {
            true -> null
            false -> true
        }
        val adjY = when (axis[1] == y2Dim - 1) {
            true -> true
            false -> null
        }
        out = MatMul.create(scope, x2, y2, MatMul.transposeA(adjX), MatMul.transposeB(adjY))
    }
    val idx: Float
    if (diff != 0) {
        if (xDim > yDim) {
            idx = (xDim + yDim - 3).toFloat()
        } else {
            idx = (xDim - 1).toFloat()
        }
        out = Squeeze.create(scope, Constant.create(scope, FloatArray(diff) { it + idx }))
    }
    if (out.asOutput().shape().numDimensions() == 1) {
        ExpandDims.create(scope, out, Constant.create(scope, 1))
    }
    return out
}
