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
import org.jetbrains.kotlinx.dl.api.core.Sequential
import java.io.File
import java.io.FileNotFoundException
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

private const val CONFIG_URL = "https://kotlindl.s3.amazonaws.com/vgg16/modelConfig.json"
private const val WEIGHTS_URL = "https://kotlindl.s3.amazonaws.com/vgg16/weights.h5"
private const val RELATIVE_CONFIG_PATH = "/vgg16/modelConfig.json"
private const val RELATIVE_WEIGHTS_PATH = "/vgg16/weights.h5"


public class VGG16ModelLoader(public val commonModelDirectory: File) {
    /** Logger for VGG16ModelLoader model. */
    private val logger: KLogger = KotlinLogging.logger {}

    init {
        if (!commonModelDirectory.exists()) throw FileNotFoundException(
            "Directory ${commonModelDirectory.name} is not found."
        )
    }

    fun loadModelRemotely(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): Sequential {
        val jsonConfigFile = getVGG16JSONConfigFile(loadingMode)
        return Sequential.loadModelConfiguration(jsonConfigFile)
    }

    fun loadClassLabelsRemotely(): MutableMap<Int, String> {

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

    public fun loadWeightsRemotely(loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS): HdfFile {
        return getVGG16WeightsFile(loadingMode)
    }

    public fun preprocessInput(floatArray: FloatArray, tensorShape: LongArray): FloatArray {
        return preprocessInput(floatArray, tensorShape, inputType = InputType.CAFFE)
    }

    /** Returns JSON file with model configuration, saved from Keras 2.x. */
    private fun getVGG16JSONConfigFile(loadingMode: LoadingMode): File {
        val dir = File(commonModelDirectory.absolutePath + "/vgg16")
        if (!dir.exists()) dir.mkdir()

        val fileName = commonModelDirectory.absolutePath + RELATIVE_CONFIG_PATH
        val file = File(fileName)

        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val `in` = URL(CONFIG_URL).openStream()
            logger.debug { "Model loading is started!" }
            Files.copy(`in`, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Model loading is finished!" }
        }

        return File(fileName)
    }

    /** Returns .h5 file with model weights, saved from Keras 2.x. */
    private fun getVGG16WeightsFile(loadingMode: LoadingMode): HdfFile {
        val fileName = commonModelDirectory.absolutePath + RELATIVE_WEIGHTS_PATH
        val file = File(fileName)
        if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
            val `in` = URL(WEIGHTS_URL).openStream()
            logger.debug { "Weights loading is started!" }
            Files.copy(`in`, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
            logger.debug { "Weights loading is finished!" }
        }

        return HdfFile(File(fileName))
    }
}



