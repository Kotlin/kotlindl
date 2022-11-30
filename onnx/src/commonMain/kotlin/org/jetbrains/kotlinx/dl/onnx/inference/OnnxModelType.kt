/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType

/**
 * Base type for [OnnxInferenceModel].
 */
public interface OnnxModelType<U : InferenceModel> : ModelType<OnnxInferenceModel, U> {
    /**
     * Shape of the input accepted by this model, without batch size.
     */
    public val inputShape: LongArray? get() = null
}
