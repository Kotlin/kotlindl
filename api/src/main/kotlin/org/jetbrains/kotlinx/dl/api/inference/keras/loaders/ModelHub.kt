/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import java.io.File
import java.nio.file.Files

private const val MODEL_CONFIG_FILE_NAME = "/modelConfig.json"
private const val WEIGHTS_FILE_NAME = "/weights.h5"
internal const val AWS_S3_URL: String = "https://kotlindl.s3.amazonaws.com"

/**
 * This model loaders provides methods for loading model, its weights and ImageNet labels (for prediction purposes) to the local directory
 * [commonModelDirectory].
 *
 * @property [commonModelDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS
 * @property [modelType] This value defines the way to S3 bucket with the model and its weights and also local directory for the model and its weights.
 *
 * @since 0.2
 */
public abstract class ModelHub(public val commonModelDirectory: File) {
    /** */
    protected val awsS3Url: String = AWS_S3_URL

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
     * @return Raw model without weights. Needs in compilation and weights loading before usage.
     */
    public abstract fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS
    ): T

    public fun <T : InferenceModel, U : InferenceModel> loadPretrainedModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS
    ): U {
        return modelType.pretrainedModel(this)
    }

    /**
     *
     */
    @Suppress("UNCHECKED_CAST")
    public operator fun <T : InferenceModel, U : InferenceModel> get(modelType: ModelType<T, U>): U {
        return loadPretrainedModel(modelType = modelType)
    }
}



