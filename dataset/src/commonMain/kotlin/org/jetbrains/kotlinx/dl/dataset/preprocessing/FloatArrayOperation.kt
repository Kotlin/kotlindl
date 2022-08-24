package org.jetbrains.kotlinx.dl.dataset.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * The aim of this class is to provide common functionality for all [Operation]s that can be applied to Pair<FloatArray, TensorShape>
 * and simplify the implementation of a new [Operation]s.
 */
public abstract class FloatArrayOperation: Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    protected abstract fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray

    override fun apply(input: Pair<FloatArray, TensorShape>): Pair<FloatArray, TensorShape> {
        val (data, shape) = input
        return applyImpl(data, shape) to getOutputShape(shape)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
}
