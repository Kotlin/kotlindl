package org.jetbrains.kotlinx.dl.api.core.layer.core

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Permutes the dimensions of the input according to a given pattern.
 * Input shape: Arbitrary
 * Output shape: Same as input shape, but with the dimensions re-ordered according to the specified pattern.
 * @property [dims] array of integers.
 * @property [name] Custom layer name.
 * @constructor Creates [Permute] object.
 */
public class Permute(
    public val dims: IntArray,
    name: String
) : Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        val outputShape = LongArray(dims.size + 1) { 0 }
        dims.forEachIndexed { i, dim ->
            val target_dim = inputShape.size(dim);
            outputShape[i + 1] = target_dim
        }
        return TensorShape(outputShape).toShape()
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val permArray = intArrayOf(0) + dims
        val perm = tf.constant(permArray)
        return tf.linalg.transpose(input, perm);
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "Permute(name=$name)"
    }
}