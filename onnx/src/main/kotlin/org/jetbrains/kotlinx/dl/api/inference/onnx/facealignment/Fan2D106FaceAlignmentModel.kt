/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

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

    /** */
    override fun close() {
        internalModel.close()
    }

    /**
     * Detects 106 [Landmark] objects for the given [imageFile].
     */
    public fun detectLandmarks(imageFile: File): List<Landmark> {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(224, 224, 3)
            }
            transformImage {
                resize {
                    outputHeight = 192
                    outputWidth = 192
                }
                convert { colorMode = ColorMode.BGR }
            }
        }

        val inputData = ONNXModels.FaceAlignment.Fan2d106.preprocessInput(preprocessing)
        val yhat = internalModel.predictRaw(inputData)

        val landMarks = mutableListOf<Landmark>()
        val floats = (yhat["fc1"] as Array<*>)[0] as FloatArray
        for (i in floats.indices step 2) {
            landMarks.add(Landmark(floats[i], floats[i + 1]))
        }

        return landMarks
    }
}

