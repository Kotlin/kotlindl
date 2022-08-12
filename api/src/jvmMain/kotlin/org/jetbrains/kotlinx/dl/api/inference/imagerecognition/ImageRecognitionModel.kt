/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTopKImageNetLabels
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * The light-weight API for solving Image Recognition task with one of the Model Hub models trained on ImageNet dataset.
 */
public class ImageRecognitionModel(
    private val internalModel: InferenceModel,
    private val modelType: ModelType<out InferenceModel, out InferenceModel>
) : InferenceModel() {
    /** Class labels for ImageNet dataset. */
    public val imageNetClassLabels: MutableMap<Int, String> = loadImageNetClassLabels()

    override val inputDimensions: LongArray
        get() = internalModel.inputDimensions

    override fun predict(inputData: FloatArray): Int {
        return internalModel.predict(inputData)
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        return internalModel.predictSoftly(inputData, predictionTensorName)
    }

    override fun reshape(vararg dims: Long) {
        throw UnsupportedOperationException("The reshape operation is not required for this model.")
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): InferenceModel {
        TODO("Not yet implemented")
    }

    /** Releases internal resources. */
    override fun close() {
        internalModel.close()
    }

    /**
     * Predicts [topK] objects for the given [imageFile].
     * Default [DataLoader] is used to load and prepare image.
     *
     * @param [imageFile] Input image [File].
     * @param [topK] Number of top ranked predictions to return
     *
     * @see preprocessData
     *
     * @return The list of pairs <label, probability> sorted from the most probable to the lowest probable.
     */
    public fun predictTopKObjects(imageFile: File, topK: Int = 5): List<Pair<String, Float>> {
        val inputData = preprocessData(imageFile)
        return predictTopKImageNetLabels(internalModel, inputData, imageNetClassLabels, topK)
    }

    /**
     * Predicts [topK] objects for the given [imageFile] with a custom [DataLoader] provided.
     *
     * @param [imageFile] Input image [File].
     * @param [dataLoader] custom [DataLoader] instance
     * @param [topK] Number of top ranked predictions to return
     *
     * @return The list of pairs <label, probability> sorted from the most probable to the lowest probable.
     */
    public fun predictTopKObjects(imageFile: File, dataLoader: DataLoader, topK: Int = 5): List<Pair<String, Float>> {
        val (inputData, _) = dataLoader.load(imageFile)
        return predictTopKImageNetLabels(internalModel, inputData, imageNetClassLabels, topK)
    }

    private fun preprocessData(imageFile: File): FloatArray {
        val (width, height) = if (modelType.channelsFirst)
            Pair(internalModel.inputDimensions[1], internalModel.inputDimensions[2])
        else
            Pair(internalModel.inputDimensions[0], internalModel.inputDimensions[1])

        val preprocessing: Preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = height.toInt()
                    outputWidth = width.toInt()
                    interpolation = InterpolationType.BILINEAR
                }
                convert { colorMode = modelType.inputColorMode }
            }
        }

        return modelType.preprocessInput(imageFile, preprocessing)
    }

    /**
     * Predicts object for the given [imageFile].
     * Default [Preprocessing] is applied to an image.
     *
     * @param [imageFile] Input image [File].
     * @see preprocessData
     *
     * @return The label of the recognized object with the highest probability.
     */
    public fun predictObject(imageFile: File): String {
        val inputData = preprocessData(imageFile)
        return imageNetClassLabels[internalModel.predict(inputData)]!!
    }

    /**
     * Predicts object for the given [imageFile] with a custom [DataLoader] provided.
     *
     * @param [imageFile] Input image [File].
     * @param [dataLoader] custom [DataLoader] instance
     *
     * @return The label of the recognized object with the highest probability.
     */
    public fun predictObject(imageFile: File, dataLoader: DataLoader): String {
        val (inputData, _) = dataLoader.load(imageFile)
        return imageNetClassLabels[internalModel.predict(inputData)]!!
    }

    public companion object {
        /**
         * Common preprocessing function for the Neural Networks trained on ImageNet and whose weights are available with the keras.application.
         *
         * It loads the data from file with the provided [dataLoader] and applied the specific preprocessing to it.
         */
        public fun ModelType<*, *>.preprocessInput(imageFile: File, dataLoader: DataLoader): FloatArray {
            val (data, shape) = dataLoader.load(imageFile)
            return preprocessInput(data, shape.dims())
        }
    }
}