package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType

/**
 * Base type for [OnnxInferenceModel].
 */
public interface OnnxModelType<T : InferenceModel, U : InferenceModel> : ModelType<T, U> {
    /**
     * Shape of the input accepted by this model, without batch size.
     */
    public val inputShape: LongArray? get() = null
}
