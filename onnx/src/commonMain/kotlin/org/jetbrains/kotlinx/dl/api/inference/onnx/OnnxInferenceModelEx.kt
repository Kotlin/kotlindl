package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider


/**
 * Convenience extension functions for inference of ONNX models using different execution providers.
 */

public inline fun <reified M : AutoCloseable, R> M.inferAndCloseUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    when (this) {
        is ExecutionProviderCompatible -> this.initializeWith(*providers)
        else -> throw IllegalArgumentException("Unsupported model type: ${M::class.simpleName}")
    }

    return this.use(block)
}

public inline fun <reified M, R> M.inferUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    when (this) {
        is ExecutionProviderCompatible -> this.initializeWith(*providers)
        else -> throw IllegalArgumentException("Unsupported model type: ${M::class.simpleName}")
    }

    return this.run(block)
}
