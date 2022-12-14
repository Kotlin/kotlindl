/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.ModelType
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val S3_FOLDER_SEPARATOR = "/"
private const val MODEL_FILE_EXTENSION = ".onnx"

/**
 * This class provides methods for loading ONNX model to the local [cacheDirectory].
 *
 * @param [cacheDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS.
 *
 * @since 0.3
 */
public class ONNXModelHub(public val cacheDirectory: File) : ModelHub() {

    /** Logger. */
    private val logger: KLogger = KotlinLogging.logger {}

    init {
        if (!cacheDirectory.exists()) {
            Files.createDirectories(cacheDirectory.toPath())
        }
    }

    /**
     * Loads a pre-trained [OnnxInferenceModel] from the ONNX model zoo.
     *
     * @param [modelType] Model type to load.
     * @param [loadingMode] Strategy of existing model use-case handling.
     * @return An instance of [OnnxInferenceModel].
     */
    @Suppress("UNCHECKED_CAST")
    public override fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode
    ): T {
        return loadModel(modelType as OnnxModelType<U>, ExecutionProvider.CPU(), loadingMode = loadingMode) as T
    }

    private fun getONNXModelFile(modelFile: String, loadingMode: LoadingMode): File {
        val fileName = cacheDirectory.absolutePath + modelFile
        val file = File(fileName)
        val parentDirectory = file.parentFile
        if (!parentDirectory.exists()) {
            Files.createDirectories(parentDirectory.toPath())
        }
        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val inputStream = URL(awsS3Url + modelFile).openStream()
            logger.debug { "Model loading is started!" }
            Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Model loading is finished!" }
        }

        return File(fileName)
    }

    /**
     * This method loads model from the ONNX model zoo corresponding to the specified [modelType].
     * The [loadingMode] parameter defines the strategy of existing model use-case handling.
     * If [loadingMode] is [LoadingMode.SKIP_LOADING_IF_EXISTS] and the model is already loaded, then the model will be loaded from the local [cacheDirectory].
     * If [loadingMode] is [LoadingMode.OVERRIDE_IF_EXISTS] the model will be overridden even if it is already loaded.
     * [executionProviders] is a list of execution providers which will be used for model inference.
     */
    public fun loadModel(
        modelType: OnnxModelType<*>,
        vararg executionProviders: ExecutionProvider,
        loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS,
    ): OnnxInferenceModel {
        val modelFile = S3_FOLDER_SEPARATOR + modelType.modelRelativePath + MODEL_FILE_EXTENSION
        val inferenceModel = OnnxInferenceModel(getONNXModelFile(modelFile, loadingMode).absolutePath)
        modelType.inputShape?.let { shape -> inferenceModel.reshape(*shape) }
        inferenceModel.initializeWith(*executionProviders)
        return inferenceModel
    }
}
