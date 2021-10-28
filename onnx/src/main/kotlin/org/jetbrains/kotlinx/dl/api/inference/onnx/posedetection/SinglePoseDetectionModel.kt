/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

public class SinglePoseDetectionModel : OnnxInferenceModel() {
    public fun detectPose(inputData: FloatArray, confidence: Float = 0.1f): DetectedPose {
        val rawPrediction = this.predictRaw(inputData)
        val rawPoseLandMarks = (rawPrediction as List<Array<Array<Array<FloatArray>>>>)[0][0][0]

        val foundPoseLandmarks = mutableListOf<PoseLandmark>()
        for (i in rawPoseLandMarks.indices) {
            val poseLandmark = PoseLandmark(
                poseLandmarkLabel = keyPoints[i]!!,
                x = rawPoseLandMarks[i][1],
                y =  rawPoseLandMarks[i][0],
                probability = rawPoseLandMarks[i][2]
            )
            foundPoseLandmarks.add(poseLandmark)
        }

        return DetectedPose(foundPoseLandmarks)
    }

    public fun detectPose(imageFile: File, confidence: Float = 0.1f): DetectedPose {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(null, null, 3)
            }
            transformImage {
                resize {
                    outputHeight = 192
                    outputWidth = 192
                }
                convert { colorMode = ColorMode.BGR }
            }
        }

        val (data, shape) = preprocessing()

        val preprocessedData = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!) // TODO: refactor to the imageShape
        )

        return this.detectPose(preprocessedData, confidence)
    }
}

// Dictionary that maps from joint names to keypoint indices.
private val keyPoints = mapOf(
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
