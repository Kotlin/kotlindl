package org.jetbrains.kotlinx.dl.dataset.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * Interface for preprocessing operations.
 * @param I is a type of an input.
 * @param O is a type of an output.
 */
public interface Operation<I, O> {
    public fun apply(input: I): O
    public fun getOutputShape(inputShape: TensorShape): TensorShape
}

/**
 * Identity operation which does nothing.
 */
public class Identity<I> : Operation<I, I> {
    override fun apply(input: I): I = input
    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
