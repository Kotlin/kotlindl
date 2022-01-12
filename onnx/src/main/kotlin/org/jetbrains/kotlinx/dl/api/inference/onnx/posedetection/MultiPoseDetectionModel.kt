/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.onnx.posedetection

import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

/**
 * MultiPoseDetectionModel is an ultra-fast and accurate model that detects 6 persons with 17 keypoints and 18 basic edges of a body for each of them.
 *
 * It internally uses [ONNXModels.PoseEstimation.MoveNetMultiPoseLighting] under the hood to make predictions.
 */
public class MultiPoseDetectionModel : OnnxInferenceModel() {
    public fun detectPoses(inputData: FloatArray, confidence: Float = 0.005f): MultiPoseDetectionResult {
        val rawPrediction = this.predictRaw(inputData)
        val rawPoseLandMarks = (rawPrediction["output_0"] as Array<Array<FloatArray>>)[0]

        val result = MultiPoseDetectionResult(mutableListOf())

        rawPoseLandMarks.forEachIndexed { poseIndex, floats ->
            val foundPoseLandmarks = mutableListOf<PoseLandmark>()

            for (i in 0..50 step 3) {
                val poseLandmark = PoseLandmark(
                    poseLandmarkLabel = keyPoints[poseIndex]!!,
                    x = floats[i + 1],
                    y = floats[i],
                    probability = floats[i + 2]
                )

                foundPoseLandmarks.add(poseLandmark)
            }

            // TODO: change order in visualisation
            // [ymin, xmin, ymax, xmax, score]
            val detectedObject = DetectedObject(
                classLabel = "person",
                probability = floats[55],
                yMin = floats[53],
                xMin = floats[52],
                yMax = floats[51],
                xMax = floats[54]
            )

            val foundPoseEdges = buildPoseEdges(foundPoseLandmarks)
            val detectedPose = DetectedPose(foundPoseLandmarks, foundPoseEdges)

            if (detectedObject.probability > confidence) result.multiplePoses.add(
                poseIndex,
                Pair(detectedObject, detectedPose)
            )
        }

        return result
    }

    public fun detectPoses(imageFile: File, confidence: Float = 0.1f): MultiPoseDetectionResult {
        val preprocessing: Preprocessing = preprocess {
            load {
                pathToData = imageFile
                imageShape = ImageShape(null, null, 3)
            }
            transformImage {
                resize {
                    outputHeight = 256
                    outputWidth = 256
                }
                convert { colorMode = ColorMode.BGR }
            }
        }

        val (data, shape) = preprocessing()

        val preprocessedData = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting.preprocessInput(
            data,
            longArrayOf(shape.width!!, shape.height!!, shape.channels!!) // TODO: refactor to the imageShape
        )

        return this.detectPoses(preprocessedData, confidence)
    }
}
