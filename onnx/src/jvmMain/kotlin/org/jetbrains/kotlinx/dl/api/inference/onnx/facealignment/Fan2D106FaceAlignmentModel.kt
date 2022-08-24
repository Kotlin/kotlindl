/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.facealignment

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel.Companion.preprocessInput
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File

private const val OUTPUT_NAME = "fc1"
private const val INPUT_SIZE = 192

/**
 * The light-weight API for solving Face Alignment task via Fan2D106 model.
 *
 * @param [internalModel] model used to make predictions
 */
public class Fan2D106FaceAlignmentModel(private val internalModel: OnnxInferenceModel) : InferenceModel by internalModel {
    /**
     * Constructs the face alignment model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String): this(OnnxInferenceModel(pathToModel))

    /**
     * Detects 106 [Landmark] objects for the given [imageFile].
     */
    public fun detectLandmarks(imageFile: File): List<Landmark> {
        val preprocessing = pipeline<BufferedImage>()
            .resize {
                outputHeight = INPUT_SIZE
                outputWidth = INPUT_SIZE
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray {}

        val inputData = ONNXModels.FaceAlignment.Fan2d106.preprocessInput(imageFile, preprocessing)
        val yhat = internalModel.predictRaw(inputData)

        val landMarks = mutableListOf<Landmark>()
        val floats = (yhat[OUTPUT_NAME] as Array<*>)[0] as FloatArray
        for (i in floats.indices step 2) {
            landMarks.add(Landmark(floats[i], floats[i + 1]))
        }

        return landMarks
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): Fan2D106FaceAlignmentModel {
        return Fan2D106FaceAlignmentModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}

