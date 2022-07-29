package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider


/**
 * Convenience extension functions for inference of ONNX models using different execution providers.
 */

public inline fun <R> OnnxInferenceModel.inferAndCloseUsing(
    vararg providers: ExecutionProvider,
    block: (OnnxInferenceModel) -> R
): R {
    this.reinitializeWith(*providers)
    return this.use(block)
}

public inline fun <R> OnnxInferenceModel.inferAndCloseUsing(
    providers: List<ExecutionProvider>,
    block: (OnnxInferenceModel) -> R
): R {
    this.reinitializeWith(*providers.toTypedArray())
    return this.use(block)
}

public inline fun <R> OnnxInferenceModel.inferUsing(
    vararg providers: ExecutionProvider,
    block: (OnnxInferenceModel) -> R
): R {
    this.reinitializeWith(*providers)
    return this.run(block)
}

public inline fun <R> OnnxInferenceModel.inferUsing(
    providers: List<ExecutionProvider>,
    block: (OnnxInferenceModel) -> R
): R {
    this.reinitializeWith(*providers.toTypedArray())
    return this.run(block)
}
