/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import android.content.Context
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider

/**
 * This class provides methods for loading ONNX model from android application resources.
 */
public class ONNXModelHub(private val context: Context) : ModelHub() {
    /**
     * Loads ONNX model from android resources.
     * By default, the model is initialized with [ExecutionProvider.CPU] execution provider.
     *
     * @param [modelType] model type from [ONNXModels]
     * @param [executionProviders] execution providers for model initialization.
     */
    public fun loadModel(
        modelType: OnnxModelType<*>,
        vararg executionProviders: ExecutionProvider = arrayOf(ExecutionProvider.CPU())
    ): OnnxInferenceModel {
        val modelResourceId = context.resources.getIdentifier(modelType.modelRelativePath, "raw", context.packageName)
        val inferenceModel = OnnxInferenceModel {
            context.resources.openRawResource(modelResourceId).use { it.readBytes() }
        }
        modelType.inputShape?.let { shape -> inferenceModel.reshape(*shape) }
        inferenceModel.initializeWith(*executionProviders)
        return inferenceModel
    }

    /**
     * It's equivalent to [loadModel] with [ExecutionProvider.CPU] execution provider.
     *
     * @param [modelType] model type from [ONNXModels]
     * @param [loadingMode] it's ignored
     */
    @Suppress("UNCHECKED_CAST")
    override fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode, /* unused */
    ): T {
        return loadModel(modelType as OnnxModelType<U>) as T
    }
}
