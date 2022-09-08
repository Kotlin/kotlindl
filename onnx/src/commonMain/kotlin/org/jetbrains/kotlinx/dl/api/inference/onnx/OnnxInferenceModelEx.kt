package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider


/**
 * Convenience extension functions for inference of ONNX models using different execution providers.
 */

public inline fun <M : ExecutionProviderCompatible, R> M.inferAndCloseUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    this.initializeWith(*providers)
    return this.use(block)
}

public inline fun <M : ExecutionProviderCompatible, R> M.inferUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    this.initializeWith(*providers)
    return this.run(block)
}
