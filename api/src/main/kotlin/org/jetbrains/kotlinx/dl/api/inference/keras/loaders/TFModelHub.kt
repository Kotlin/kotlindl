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
 * This class  provides methods for loading Keras models, its weights and ImageNet labels (for prediction purposes) to the local [cacheDirectory].
 *
 * @param [cacheDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS.
 * @since 0.2
 */
public class TFModelHub(cacheDirectory: File) : ModelHub(cacheDirectory) {
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
    @Suppress("UNCHECKED_CAST")
    public override fun <T : InferenceModel, U : InferenceModel> loadModel(
        modelType: ModelType<T, U>,
        loadingMode: LoadingMode
    ): T {
        val jsonConfigFile = if (modelType is TFModels.CV) {
            getJSONConfigFile(modelType, loadingMode, modelType.noTop)
        } else {
            getJSONConfigFile(modelType, loadingMode)
        }

        return when (modelType) {
            is TFModels.CV.VGG16 -> freezeAllLayers(
                Sequential.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.VGG19 -> freezeAllLayers(
                Sequential.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet18 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet34 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet50 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet101 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet152 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet50v2 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet101v2 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.ResNet152v2 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.MobileNet -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.MobileNetV2 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.Inception -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.Xception -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.DenseNet121 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.DenseNet169 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.DenseNet201 -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.NASNetMobile -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            is TFModels.CV.NASNetLarge -> freezeAllLayers(
                Functional.loadModelConfiguration(
                    jsonConfigFile,
                    modelType.inputShape
                )
            ) as T
            else -> TODO()
        }
    }

    private fun freezeAllLayers(model: GraphTrainableModel): GraphTrainableModel {
        for (layer in model.layers) {
            layer.isTrainable = false
        }
        return model
    }

    /** Forms mapping of class label to class name for the ImageNet dataset. */
    public fun loadClassLabels(): MutableMap<Int, String> {
        val pathToIndices = "/datasets/vgg/imagenet_class_index.json"

        fun parse(name: String): Any? {
            val cls = Parser::class.java
            return cls.getResourceAsStream(name)?.let { inputStream ->
                return Parser.default().parse(inputStream, Charsets.UTF_8)
            }
        }

        val classIndices = parse(pathToIndices) as JsonObject

        val imageNetClassIndices = mutableMapOf<Int, String>()

        for (key in classIndices.keys) {
            imageNetClassIndices[key.toInt()] = (classIndices[key] as JsonArray<*>)[1].toString()
        }
        return imageNetClassIndices
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
        val noTop = if (modelType is TFModels.CV) modelType.noTop else false
        return getWeightsFile(modelType, loadingMode, noTop)
    }

    /** Returns JSON file with model configuration, saved from Keras 2.x. */
    private fun getJSONConfigFile(modelType: ModelType<*, *>, loadingMode: LoadingMode, noTop: Boolean = false): File {
        var modelDirectory = "/" + modelType.modelRelativePath
        if(noTop) modelDirectory += "/notop"

        val relativeConfigPath = modelDirectory + MODEL_CONFIG_FILE_NAME
        val configURL = AWS_S3_URL + modelDirectory + MODEL_CONFIG_FILE_NAME

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

    /** Returns .h5 file with model weights, saved from Keras 2.x. */
    private fun getWeightsFile(modelType: ModelType<*, *>, loadingMode: LoadingMode, noTop: Boolean = false): HdfFile {
        var modelDirectory = "/" + modelType.modelRelativePath
        if(noTop) modelDirectory += "/notop"

        val relativeWeightsPath = modelDirectory + WEIGHTS_FILE_NAME
        val weightsURL = AWS_S3_URL + modelDirectory + WEIGHTS_FILE_NAME

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
}




