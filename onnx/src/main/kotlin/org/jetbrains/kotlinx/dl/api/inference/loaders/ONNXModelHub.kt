/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

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
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
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
public class ONNXModelHub(commonModelDirectory: File, modelType: ModelType) :
    ModelHub(commonModelDirectory, modelType) {
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
    public override fun loadModel(loadingMode: LoadingMode): InferenceModel {
        val jsonConfigFile = getJSONConfigFile(loadingMode)
        return when (modelType) {
            ONNXModels.CV.VGG_16 -> Sequential.loadModelConfiguration(jsonConfigFile)
            TFModels.CV.VGG_19 -> Sequential.loadModelConfiguration(jsonConfigFile)
            TFModels.CV.ResNet_18 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_34 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_50 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_101 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_152 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_50_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_101_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.ResNet_151_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.MobileNet -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.MobileNetv2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.Inception -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.Xception -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.DenseNet121 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.DenseNet169 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.DenseNet201 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.NASNetMobile -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            TFModels.CV.NASNetLarge -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            else -> TODO()
        }
    }
}




