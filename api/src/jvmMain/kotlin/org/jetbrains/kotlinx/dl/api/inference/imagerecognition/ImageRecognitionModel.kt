/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.imagerecognition

import com.beust.klaxon.JsonArray
import com.beust.klaxon.JsonObject
import com.beust.klaxon.Parser
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

/**
 * The light-weight API for solving Image Recognition task with one of the Model Hub models trained on ImageNet dataset.
 */
public class ImageRecognitionModel(
    internalModel: InferenceModel,
    private val modelType: ModelType<out InferenceModel, out InferenceModel>
) : ImageRecognitionModelBase<BufferedImage>(internalModel) {
    /** Class labels for ImageNet dataset. */
    override val classLabels: Map<Int, String> = loadImageNetClassLabels()

    override val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
        get() {
            val (width, height) = if (modelType.channelsFirst)
                Pair(internalModel.inputDimensions[1], internalModel.inputDimensions[2])
            else
                Pair(internalModel.inputDimensions[0], internalModel.inputDimensions[1])

            return pipeline<BufferedImage>()
                .resize {
                    outputHeight = height.toInt()
                    outputWidth = width.toInt()
                    interpolation = InterpolationType.BILINEAR
                }
                .convert { colorMode = modelType.inputColorMode }
                .toFloatArray {}
                .call(modelType.preprocessor)
        }

    /**
     * Predicts [topK] objects for the given [imageFile].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [imageFile] Input image [File].
     * @param [topK] Number of top ranked predictions to return
     *
     * @see preprocessing
     *
     * @return The list of pairs <label, probability> sorted from the most probable to the lowest probable.
     */
    @Throws(IOException::class)
    public fun predictTopKObjects(imageFile: File, topK: Int): List<Pair<String, Float>> {
        return predictTopKObjects(ImageConverter.toBufferedImage(imageFile), topK)
    }

    /**
     * Predicts object for the given [imageFile].
     * Default preprocessing [Operation] is applied to an image.
     *
     * @param [imageFile] Input image [File].
     * @see preprocessing
     *
     * @return The label of the recognized object with the highest probability.
     */
    @Throws(IOException::class)
    public fun predictObject(imageFile: File): String {
        return predictObject(ImageConverter.toBufferedImage(imageFile))
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): ImageRecognitionModel {
        return ImageRecognitionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights), modelType)
    }
}

/** Forms mapping of class label to class name for the ImageNet dataset. */
public fun loadImageNetClassLabels(): Map<Int, String> {
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