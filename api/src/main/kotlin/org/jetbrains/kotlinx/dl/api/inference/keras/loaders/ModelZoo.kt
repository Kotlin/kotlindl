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
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val MODEL_CONFIG_FILE_NAME = "/modelConfig.json"
private const val WEIGHTS_FILE_NAME = "/weights.h5"
internal const val AWS_S3_URL = "https://kotlindl.s3.amazonaws.com"

/**
 * This model loaders provides methods for loading model, its weights and ImageNet labels (for prediction purposes) to the local directory
 * [commonModelDirectory].
 *
 * @property [commonModelDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS
 * @property [modelType] This value defines the way to S3 bucket with the model and its weights and also local directory for the model and its weights.
 *
 * @since 0.2
 */
public class ModelZoo(public val commonModelDirectory: File, public val modelType: ModelType) {
    private val modelDirectory = "/" + modelType.modelName
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
    public fun loadModel(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): GraphTrainableModel {
        val jsonConfigFile = getJSONConfigFile(loadingMode)
        return when (modelType) {
            Models.TensorFlow.VGG_16 -> Sequential.loadModelConfiguration(jsonConfigFile)
            Models.TensorFlow.VGG_19 -> Sequential.loadModelConfiguration(jsonConfigFile)
            Models.TensorFlow.ResNet_18 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_34 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_50 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_101 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_152 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_50_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_101_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.ResNet_151_v2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.MobileNet -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.MobileNetv2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.Inception -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.Xception -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.DenseNet121 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.DenseNet169 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.DenseNet201 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.NASNetMobile -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            Models.TensorFlow.NASNetLarge -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            else -> TODO()
        }
    }

    private fun freezeAllLayers(model: Functional): GraphTrainableModel {
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
     * @param [loadingMode] Strategy of existing model use-case handling.
     * @return Compiled model with initialized weights.
     */
    public fun loadWeights(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): HdfFile {
        return getWeightsFile(loadingMode)
    }

    /**
     * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
     *
     * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according given [modelType].
     */
    public fun preprocessInput(data: FloatArray, tensorShape: LongArray): FloatArray {
        return preprocessInput(data, tensorShape, modelType)
    }

    /** Returns JSON file with model configuration, saved from Keras 2.x. */
    private fun getJSONConfigFile(loadingMode: LoadingMode): File {
        val dir = File(commonModelDirectory.absolutePath + modelDirectory)
        if (!dir.exists()) dir.mkdir()

        val fileName = commonModelDirectory.absolutePath + relativeConfigPath
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
    private fun getWeightsFile(loadingMode: LoadingMode): HdfFile {
        val fileName = commonModelDirectory.absolutePath + relativeWeightsPath
        val file = File(fileName)
        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val inputStream = URL(weightsURL).openStream()
            logger.debug { "Weights loading is started!" }
            Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Weights loading is finished!" }
        }

        return HdfFile(File(fileName))
    }
}

/**
 * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
 *
 * It takes [data] as input with shape [tensorShape] and applied the specific preprocessing according chosen [modelType].
 */
public fun preprocessInput(data: FloatArray, tensorShape: LongArray, modelType: ModelType): FloatArray {
    return when (modelType) {
        Models.TensorFlow.VGG_16 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.VGG_19 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_18 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_34 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_50 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_101 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_152 -> preprocessInput(data, tensorShape, inputType = InputType.CAFFE)
        Models.TensorFlow.ResNet_50_v2 -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.ResNet_101_v2 -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.ResNet_151_v2 -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.MobileNet -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.MobileNetv2 -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.Inception -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.Xception -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.DenseNet121 -> preprocessInput(data, tensorShape, inputType = InputType.TORCH)
        Models.TensorFlow.DenseNet169 -> preprocessInput(data, tensorShape, inputType = InputType.TORCH)
        Models.TensorFlow.DenseNet201 -> preprocessInput(data, tensorShape, inputType = InputType.TORCH)
        Models.TensorFlow.NASNetMobile -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        Models.TensorFlow.NASNetLarge -> preprocessInput(data, tensorShape, inputType = InputType.TF)
        else -> TODO()
    }
}



