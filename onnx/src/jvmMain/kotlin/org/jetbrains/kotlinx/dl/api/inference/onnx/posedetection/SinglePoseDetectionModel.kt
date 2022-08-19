/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseEdge
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.fileLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File
import java.lang.Float.min

private const val OUTPUT_NAME = "output_0"

/**
 * SinglePoseDetectionModel is an ultra-fast and accurate model that detects 17 keypoints and 18 basic edges of a body.
 *
 * It internally uses [ONNXModels.PoseDetection.MoveNetSinglePoseLighting]
 * or [ONNXModels.PoseDetection.MoveNetSinglePoseThunder] under the hood to make predictions.
 *
 * @param internalModel model used to make predictions
 */
public class SinglePoseDetectionModel(private val internalModel: OnnxInferenceModel) : InferenceModel by internalModel {
    /**
     * Constructs the pose detection model from a given path.
     * @param [pathToModel] path to model
     */
    public constructor(pathToModel: String): this(OnnxInferenceModel(pathToModel))

    public fun detectPose(inputData: FloatArray): DetectedPose {
        val rawPrediction = internalModel.predictRaw(inputData)
        val rawPoseLandMarks = (rawPrediction[OUTPUT_NAME] as Array<Array<Array<FloatArray>>>)[0][0]

        val foundPoseLandmarks = mutableListOf<PoseLandmark>()
        for (i in rawPoseLandMarks.indices) {
            val poseLandmark = PoseLandmark(
                poseLandmarkLabel = keyPoints[i]!!,
                x = rawPoseLandMarks[i][1],
                y = rawPoseLandMarks[i][0],
                probability = rawPoseLandMarks[i][2]
            )
            foundPoseLandmarks.add(i, poseLandmark)
        }

        val foundPoseEdges = buildPoseEdges(foundPoseLandmarks)

        return DetectedPose(foundPoseLandmarks, foundPoseEdges)
    }

    public fun detectPose(imageFile: File): DetectedPose {
        val height = inputDimensions[0]
        val width = inputDimensions[1]

        val preprocessing = pipeline<BufferedImage>()
            .resize {
                    outputHeight = height.toInt()
                    outputWidth = width.toInt()
                }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray {  }
            .call(ONNXModels.PoseDetection.MoveNetSinglePoseLighting.preprocessor)

        val data = preprocessing.fileLoader().load(imageFile).first

        return this.detectPose(data)
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean,
        copyWeights: Boolean
    ): SinglePoseDetectionModel {
        return SinglePoseDetectionModel(internalModel.copy(copiedModelName, saveOptimizerState, copyWeights))
    }
}

internal fun buildPoseEdges(foundPoseLandmarks: List<PoseLandmark>): List<PoseEdge> {
    val foundPoseEdges = mutableListOf<PoseEdge>()
    edgeKeyPointsPairs.forEach {
        val startPoint = foundPoseLandmarks[it.first]
        val endPoint = foundPoseLandmarks[it.second]
        foundPoseEdges.add(
            PoseEdge(
                poseEdgeLabel = startPoint.poseLandmarkLabel + "_" + endPoint.poseLandmarkLabel,
                probability = min(startPoint.probability, endPoint.probability),
                start = startPoint,
                end = endPoint
            )
        )
    }
    return foundPoseEdges
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
