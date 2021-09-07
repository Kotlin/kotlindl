/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.SSDObjectDetectionModel
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

/**
 * This model loaders provides methods for loading model, its weights and ImageNet labels (for prediction purposes) to the local directory
 * [commonModelDirectory].
 *
 * @property [commonModelDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS
 * @property [modelType] This value defines the way to S3 bucket with the model and its weights and also local directory for the model and its weights.
 *
 * @since 0.3
 */
public class ONNXModelHub(commonModelDirectory: File, modelType: ModelType) :
    ModelHub(commonModelDirectory, modelType) {
    private val modelFile = "/" + modelType.modelRelativePath + ".onnx"

    /** Logger. */
    private val logger: KLogger = KotlinLogging.logger {}

    init {
        if (!commonModelDirectory.exists()) {
            Files.createDirectories(commonModelDirectory.toPath())
        }
    }

    /**
     * Loads model configuration without weights.
     *
     * @param [loadingMode] Strategy of existing model use-case handling.
     * @return An example of [OnnxInferenceModel].
     */
    public override fun loadModel(loadingMode: LoadingMode): InferenceModel {
        val inferenceModel = if (modelType == ONNXModels.ObjectDetection.SSD) {
            SSDObjectDetectionModel()
        } else {
            OnnxInferenceModel()
        }
        return OnnxInferenceModel.initializeONNXModel(inferenceModel, getONNXModelFile(loadingMode).absolutePath)
    }

    private fun getONNXModelFile(loadingMode: LoadingMode): File {
        val fileName = commonModelDirectory.absolutePath + modelFile
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
}




