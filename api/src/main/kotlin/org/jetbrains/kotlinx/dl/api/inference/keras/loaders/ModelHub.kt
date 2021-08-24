/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import io.jhdf.HdfFile
import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val MODEL_CONFIG_FILE_NAME = "/modelConfig.json"
private const val WEIGHTS_FILE_NAME = "/weights.h5"


/**
 * This model loaders provides methods for loading model, its weights and ImageNet labels (for prediction purposes) to the local directory
 * [commonModelDirectory].
 *
 * @property [commonModelDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS
 * @property [modelType] This value defines the way to S3 bucket with the model and its weights and also local directory for the model and its weights.
 *
 * @since 0.2
 */
public abstract class ModelHub(public val commonModelDirectory: File, public val modelType: ModelType) {
    protected val AWS_S3_URL: String = "https://kotlindl.s3.amazonaws.com"
    private val modelDirectory = "/" + modelType.modelRelativePath
    private val relativeConfigPath = modelDirectory + MODEL_CONFIG_FILE_NAME
    private val relativeWeightsPath = modelDirectory + WEIGHTS_FILE_NAME
    private val configURL = AWS_S3_URL + modelDirectory + MODEL_CONFIG_FILE_NAME
    private val weightsURL = AWS_S3_URL + modelDirectory + WEIGHTS_FILE_NAME

    /** Logger for modelZoo model. */
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
     * @return Raw model without weights. Needs in compilation and weights loading via [loadWeights] before usage.
     */
    public abstract fun loadModel(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): InferenceModel


    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according given [modelType].
     */
    public fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
        return preprocessInput(data, tensorShape, modelType)
    }
}


public fun preprocessInput(data: FloatArray, tensorShape: LongArray, modelType: ModelType): FloatArray {
    return modelType.preprocessInput(data, tensorShape)
}



