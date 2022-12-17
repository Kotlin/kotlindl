/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

import io.jhdf.HdfFile
import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val MODEL_CONFIG_FILE_NAME = "/${GraphTrainableModel.MODEL_CONFIG_JSON}"
private const val WEIGHTS_FILE_NAME = "/weights.h5"

/**
 * This class  provides methods for loading Keras models, its weights and ImageNet labels (for prediction purposes) to the local [cacheDirectory].
 *
 * @param [cacheDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS.
 * @since 0.2
 *
 * @see TFModelType
 * @see TFModels
 */
public class TFModelHub(public val cacheDirectory: File) : ModelHub() {
    /** Logger for modelZoo model. */
    private val logger: KLogger = KotlinLogging.logger {}

    init {
        if (!cacheDirectory.exists()) {
            Files.createDirectories(cacheDirectory.toPath())
        }
    }

    /**
     * Loads model configuration without weights.
     *
     * @param [modelType] This unique identifier defines the way to the S3 bucket with the model and its weights and the local directory for the model and its weights.
     * @param [loadingMode] Strategy of existing model use-case handling.
     * @return Raw model without weights. Needs in compilation and weights loading via [loadWeights] before usage.
     */
    public override fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode
    ): T {
        val jsonConfigFile = getJSONConfigFile(modelType, loadingMode)
        return (modelType as TFModelType).loadModelConfiguration(jsonConfigFile)
    }

    /** Forms mapping of class label to class name for the ImageNet dataset. */
    public fun loadClassLabels(): Map<Int, String> {
        return Imagenet.V1k.labels()
    }

    /**
     * Loads model weights.
     *
     * @param [modelType] This unique identifier defines the way to the S3 bucket with the model and its weights and the local directory for the model and its weights.
     * @param [loadingMode] Strategy of existing model use-case handling.
     * @return Compiled model with initialized weights.
     */
    public fun loadWeights(
        modelType: ModelType<*, *>,
        loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS
    ): HdfFile {
        val modelDirectory = "/" + modelType.modelRelativePath
        val relativeWeightsPath = modelDirectory + WEIGHTS_FILE_NAME
        val weightsURL = awsS3Url + modelDirectory + WEIGHTS_FILE_NAME
        val fileName = cacheDirectory.absolutePath + relativeWeightsPath
        val file = File(fileName)
        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val inputStream = URL(weightsURL).openStream()
            logger.info { "Weights loading is started!" }
            Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.info { "Weights loading is finished!" }
        }
        return HdfFile(File(fileName))
    }

    /** Returns JSON file with model configuration, saved from Keras 2.x. */
    private fun getJSONConfigFile(modelType: ModelType<*, *>, loadingMode: LoadingMode): File {
        val modelDirectory = "/" + modelType.modelRelativePath
        val relativeConfigPath = modelDirectory + MODEL_CONFIG_FILE_NAME
        val configURL = awsS3Url + modelDirectory + MODEL_CONFIG_FILE_NAME

        val dir = File(cacheDirectory.absolutePath + modelDirectory)
        if (!dir.exists()) Files.createDirectories(dir.toPath())

        val fileName = cacheDirectory.absolutePath + relativeConfigPath
        val file = File(fileName)

        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val inputStream = URL(configURL).openStream()
            logger.debug { "Model loading is started!" }
            Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Model loading is finished!" }
        }

        return File(fileName)
    }
}




