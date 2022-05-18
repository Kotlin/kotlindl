/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

private const val OUTPUT_NAME = "fc1"
private const val INPUT_SIZE = 192

/**
 * The light-weight API for solving Face Alignment task via Fan2D106 model.
 */
public class Fan2D106FaceAlignmentModel(private val internalModel: OnnxInferenceModel) : InferenceModel() {
    override val inputDimensions: LongArray
        get() = internalModel.inputDimensions

    override fun predict(inputData: FloatArray): Int {
        return internalModel.predict(inputData)
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        return internalModel.predictSoftly(inputData, predictionTensorName)
    }

    override fun reshape(vararg dims: Long) {
        TODO()
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): TensorFlowInferenceModel {
        TODO("Not yet implemented")
    }

    /** Releases the model resources. */
    override fun close() {
        internalModel.close()
    }

    /**
     * Detects 106 [Landmark] objects for the given [imageFile].
     */
    public fun detectLandmarks(imageFile: File): List<Landmark> {
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = INPUT_SIZE
                    outputWidth = INPUT_SIZE
                }
                convert { colorMode = ColorMode.BGR }
            }
        }

        val inputData = ONNXModels.FaceAlignment.Fan2d106.preprocessInput(imageFile, preprocessing)
        val yhat = internalModel.predictRaw(inputData)

        val landMarks = mutableListOf<Landmark>()
        val floats = (yhat[OUTPUT_NAME] as Array<*>)[0] as FloatArray
        for (i in floats.indices step 2) {
            landMarks.add(Landmark(floats[i], floats[i + 1]))
        }

        return landMarks
    }
}

