/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.facealignment

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.facealignment.Landmark
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import java.awt.image.BufferedImage
import java.io.File
import java.io.IOException

private const val OUTPUT_NAME = "fc1"

/**
 * The light-weight API for solving Face Alignment task via Fan2D106 model.
 *
 * @param [internalModel] model used to make predictions
 */
public class Fan2D106FaceAlignmentModel(override val internalModel: OnnxInferenceModel) :
    FaceAlignmentModelBase<BufferedImage>(), InferenceModel by internalModel {

    override val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
        get() = pipeline<BufferedImage>()
            .resize {
                outputWidth = internalModel.inputDimensions[2].toInt()
                outputHeight = internalModel.inputDimensions[1].toInt()
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray {}
            .call(ONNXModels.FaceAlignment.Fan2d106.preprocessor)

    override val outputName: String = OUTPUT_NAME

    /**
     * Constructs the face alignment model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String) : this(OnnxInferenceModel(pathToModel))

    /**
     * Detects 106 [Landmark] objects for the given [imageFile].
     * @param [imageFile] file containing an input image
     */
    @Throws(IOException::class)
    public fun detectLandmarks(imageFile: File): List<Landmark> {
        return detectLandmarks(ImageConverter.toBufferedImage(imageFile))
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): Fan2D106FaceAlignmentModel {
        return Fan2D106FaceAlignmentModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}

