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
import java.io.FileNotFoundException
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val MODEL_CONFIG_FILE_NAME = "/modelConfig.json"
private const val WEIGHTS_FILE_NAME = "/weights.h5"
private const val AWS_S3_URL = "https://kotlindl.s3.amazonaws.com"

/**
 * This model loaders provides methods for loading model, its weights and ImageNet labels (for prediction purposes) to the local directory
 * [commonModelDirectory].
 *
 * @property [commonModelDirectory] The directory for all loaded models. It should be created before model loading and should have all required permissions for file writing/reading on your OS
 * @property [modelType] This value defines the way to S3 bucket with the model and its weights and also local directory for the model and its weights.
 */
public class ModelLoader(public val commonModelDirectory: File, public val modelType: ModelType) {
    private val modelDirectory = "/" + modelType.modelName
    private val relativeConfigPath = modelDirectory + MODEL_CONFIG_FILE_NAME
    private val relativeWeightsPath = modelDirectory + WEIGHTS_FILE_NAME
    private val configURL = AWS_S3_URL + modelDirectory + MODEL_CONFIG_FILE_NAME
    private val weightsURL = AWS_S3_URL + modelDirectory + WEIGHTS_FILE_NAME

    /** Logger for ModelLoader model. */
    private val logger: KLogger = KotlinLogging.logger {}

    init {
        if (!commonModelDirectory.exists()) throw FileNotFoundException(
            "Directory ${commonModelDirectory.name} is not found."
        )
    }

    public fun loadModel(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): GraphTrainableModel {
        val jsonConfigFile = getJSONConfigFile(loadingMode)
        return when (modelType) {
            ModelType.VGG_16 -> Sequential.loadModelConfiguration(jsonConfigFile)
            ModelType.VGG_19 -> Sequential.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_50 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_101 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_151 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_50_v2 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_101_v2 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.ResNet_151_v2 -> Functional.loadModelConfiguration(jsonConfigFile)
            ModelType.MobileNet -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
            ModelType.MobileNetv2 -> freezeAllLayers(Functional.loadModelConfiguration(jsonConfigFile))
        }
    }

    private fun freezeAllLayers(model: Functional): GraphTrainableModel {
        for (layer in model.layers) {
            layer.isTrainable = false
        }
        return model
    }

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

    public fun loadWeights(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): HdfFile {
        return getWeightsFile(loadingMode)
    }

    public fun preprocessInput(floatArray: FloatArray, tensorShape: LongArray): FloatArray {
        return when (modelType) {
            ModelType.VGG_16 -> preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            ModelType.VGG_19 -> preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            ModelType.ResNet_50 -> preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            ModelType.ResNet_101 -> preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            ModelType.ResNet_151 -> preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
            ModelType.ResNet_50_v2 -> preprocessInput(floatArray, inputType = InputType.TF)
            ModelType.ResNet_101_v2 -> preprocessInput(floatArray, inputType = InputType.TF)
            ModelType.ResNet_151_v2 -> preprocessInput(floatArray, inputType = InputType.TF)
            ModelType.MobileNet -> preprocessInput(floatArray, inputType = InputType.TF)
            ModelType.MobileNetv2 -> preprocessInput(floatArray, inputType = InputType.TF)
        }
    }

    /** Returns JSON file with model configuration, saved from Keras 2.x. */
    private fun getJSONConfigFile(loadingMode: LoadingMode): File {
        val dir = File(commonModelDirectory.absolutePath + modelDirectory)
        if (!dir.exists()) dir.mkdir()

        val fileName = commonModelDirectory.absolutePath + relativeConfigPath
        val file = File(fileName)

        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val `in` = URL(configURL).openStream()
            logger.debug { "Model loading is started!" }
            Files.copy(`in`, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Model loading is finished!" }
        }

        return File(fileName)
    }

    /** Returns .h5 file with model weights, saved from Keras 2.x. */
    private fun getWeightsFile(loadingMode: LoadingMode): HdfFile {
        val fileName = commonModelDirectory.absolutePath + relativeWeightsPath
        val file = File(fileName)
        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val `in` = URL(weightsURL).openStream()
            logger.debug { "Weights loading is started!" }
            Files.copy(`in`, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Weights loading is finished!" }
        }

        return HdfFile(File(fileName))
    }
}



