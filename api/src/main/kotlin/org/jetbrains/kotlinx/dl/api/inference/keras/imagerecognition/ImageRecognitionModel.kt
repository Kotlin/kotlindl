/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.imagerecognition

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTopKImageNetLabels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import java.io.File

public class ImageRecognitionModel(
    private val internalModel: GraphTrainableModel,
    private val modelType: ModelType<out InferenceModel, out InferenceModel>
) : InferenceModel() {
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
        TODO("Not yet implemented")
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): TensorFlowInferenceModel {
        TODO("Not yet implemented")
    }

    override fun close() {
        internalModel.close()
    }

    public fun predictTopKObjects(imageFile: File, topK: Int = 5): MutableMap<Int, Pair<String, Float>> {
        val inputData = preprocessData(imageFile)
        return predictTopKImageNetLabels(internalModel, inputData, imageNetClassLabels, topK)
    }

    private fun preprocessData(imageFile: File): FloatArray {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(224, 224, 3) // TODO: it should be empty or became a parameter
                colorMode = ColorOrder.BGR
            }
        }

        val inputData = modelType.preprocessInput(preprocessing().first, inputDimensions)
        return inputData
    }

    public fun predictObject(imageFile: File): String {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(224, 224, 3) // TODO: it should be empty or became a parameter
                colorMode = ColorOrder.BGR
            }
        }

        val inputData = modelType.preprocessInput(preprocessing().first, inputDimensions)
        return imageNetClassLabels[internalModel.predict(inputData)]!!
    }
}
