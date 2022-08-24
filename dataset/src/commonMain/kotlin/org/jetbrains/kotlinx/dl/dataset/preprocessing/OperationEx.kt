package org.jetbrains.kotlinx.dl.dataset.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * Convenience functions for executing custom logic after applying [Operation].
 * Could be useful for debugging purposes.
 */
public fun <I, O> Operation<I,O>.onResult(block: (O) -> Unit): Operation<I, O> {
    return PreprocessingPipeline(this, object : Operation<O, O> {
        override fun apply(input: O): O {
            try {
                block(input)
            } finally {
                return input
            }
        }
        override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
    })
}
