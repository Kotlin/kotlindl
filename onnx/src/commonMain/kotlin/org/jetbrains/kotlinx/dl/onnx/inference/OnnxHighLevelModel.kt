/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import ai.onnxruntime.OrtSession
import org.jetbrains.kotlinx.dl.api.summary.ModelHubModelSummary
import org.jetbrains.kotlinx.dl.api.summary.ModelWithSummary
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.summary.ModelSummary
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider

/**
 * Base class for pre-trained high-level onnx models.
 *
 * @param [I] input type
 * @param [R] output type
 */
public interface OnnxHighLevelModel<I, R> : ExecutionProviderCompatible, ModelWithSummary {
    /**
     * Model used to make predictions.
     */
    public val internalModel: OnnxInferenceModel

    /**
     * Preprocessing operation specific to this model.
     */
    public val preprocessing: Operation<I, Pair<FloatArray, TensorShape>>

    /**
     * Corresponding [OnnxModelType] from [ONNXModels].
     * Used for model summary.
     * Should be null for custom models.
     */
    public val type: OnnxModelType<*, *>?

    /**
     * Converts raw model output to the result.
     */
    public fun convert(output: OrtSession.Result): R

    /**
     * Makes prediction on the given [input].
     */
    public fun predict(input: I): R {
        val preprocessedInput = preprocessing.apply(input)
        return internalModel.predictRaw(preprocessedInput.first) { convert(it) }
    }

    override fun initializeWith(vararg executionProviders: ExecutionProvider) {
        internalModel.initializeWith(*executionProviders)
    }

    override fun summary(): ModelSummary {
        return ModelHubModelSummary(internalModel.summary(), type)
    }
}
