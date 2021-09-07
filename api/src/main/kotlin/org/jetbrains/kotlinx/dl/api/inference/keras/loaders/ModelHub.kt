/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

import mu.KLogger
import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
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
public abstract class ModelHub(public val commonModelDirectory: File, public val modelType: ModelType) {
    /** */
    protected val awsS3Url: String = AWS_S3_URL
    private val modelDirectory = "/" + modelType.modelRelativePath

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
    public abstract fun loadModel(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): InferenceModel


    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according given [modelType].
     */
    public fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
        return preprocessInput(data, tensorShape, modelType)
    }

    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes preprocessing pipeline, invoke it and applied the specific preprocessing according given [modelType].
     */
    public fun preprocessInput(preprocessing: Preprocessing): FloatArray {
        val (data, shape) = preprocessing()
        return modelType.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels)
        ) // TODO: need to be 4 or 3 in all cases
    }
}

/** Wraps the [ModelType.preprocessInput] functionality. */
public fun preprocessInput(data: FloatArray, tensorShape: LongArray, modelType: ModelType): FloatArray {
    return modelType.preprocessInput(data, tensorShape)
}



