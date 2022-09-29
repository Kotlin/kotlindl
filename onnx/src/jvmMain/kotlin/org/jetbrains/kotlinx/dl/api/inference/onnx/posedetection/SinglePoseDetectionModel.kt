/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
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


/**
 * SinglePoseDetectionModel is an ultra-fast and accurate model that detects 17 keypoints and 18 basic edges of a body.
 *
 * It internally uses [ONNXModels.PoseDetection.MoveNetSinglePoseLighting]
 * or [ONNXModels.PoseDetection.MoveNetSinglePoseThunder] under the hood to make predictions.
 *
 * @param internalModel model used to make predictions
 */
public class SinglePoseDetectionModel(override val internalModel: OnnxInferenceModel) :
    SinglePoseDetectionModelBase<BufferedImage>(), InferenceModel by internalModel {

    override val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
        get() = pipeline<BufferedImage>()
            .resize {
                outputHeight = inputDimensions[0].toInt()
                outputWidth = inputDimensions[1].toInt()
            }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
            .call(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.preprocessor)

    /**
     * Constructs the pose detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String) : this(OnnxInferenceModel(pathToModel))

    /**
     * Detects a pose for the given [imageFile].
     * @param [imageFile] file containing an input image
     */
    @Throws(IOException::class)
    public fun detectPose(imageFile: File): DetectedPose {
        return detectPose(ImageConverter.toBufferedImage(imageFile))
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): SinglePoseDetectionModel {
        return SinglePoseDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}
