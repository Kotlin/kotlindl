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

private const val OUTPUT_NAME = "output_0"

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

    override val outputName: String = OUTPUT_NAME
    override val keyPointsLabels: Map<Int, String> = keyPoints
    override val edgeKeyPoints: List<Pair<Int, Int>> = edgeKeyPointsPairs

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

/**
 * Dictionary that maps from joint names to keypoint indices.
 */
public val keyPoints: Map<Int, String> = mapOf(
    0 to "nose",
    1 to "left_eye",
    2 to "right_eye",
    3 to "left_ear",
    4 to "right_ear",
    5 to "left_shoulder",
    6 to "right_shoulder",
    7 to "left_elbow",
    8 to "right_elbow",
    9 to "left_wrist",
    10 to "right_wrist",
    11 to "left_hip",
    12 to "right_hip",
    13 to "left_knee",
    14 to "right_knee",
    15 to "left_ankle",
    16 to "right_ankle"
)

/**
 * Pair of points which define body edges.
 */
public val edgeKeyPointsPairs: List<Pair<Int, Int>> = listOf(
    Pair(0, 1),
    Pair(0, 2),
    Pair(1, 3),
    Pair(2, 4),
    Pair(0, 5),
    Pair(0, 6),
    Pair(5, 7),
    Pair(7, 9),
    Pair(6, 8),
    Pair(8, 10),
    Pair(5, 6),
    Pair(5, 11),
    Pair(6, 12),
    Pair(11, 12),
    Pair(11, 13),
    Pair(13, 15),
    Pair(12, 14),
    Pair(14, 16)
)
