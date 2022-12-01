/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.singlepose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedPosePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to each image before prediction.
 */
fun poseDetectionMoveNet() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseDetection.MoveNetSinglePoseLighting
    val model = modelHub.loadModel(modelType)
    model.printSummary()

    model.use {
        println(it)

        val file = getFileFromResource("datasets/poses/single/3.jpg")
        val image = ImageConverter.toBufferedImage(file)
        val preprocessing = pipeline<BufferedImage>()
            .resize {
                outputHeight = 192
                outputWidth = 192
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)

        val inputData = preprocessing.apply(image)

        val rawPoseLandMarks = it.predictRaw(inputData) { result ->
            result.get2DFloatArray("output_0")
        }
        println(rawPoseLandMarks.contentDeepToString())

        // Dictionary that maps from joint names to keypoint indices.
        val keypoints = mapOf(
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

        rawPoseLandMarks.forEachIndexed { index, data ->
            println(keypoints[index] + " x = " + data[1] + " y =  " + data[0] + " score = " + data[2])
        }

        val foundPoseLandmarks = mutableListOf<PoseLandmark>()
        for (i in rawPoseLandMarks.indices) {
            val poseLandmark = PoseLandmark(
                x = rawPoseLandMarks[i][1],
                y = rawPoseLandMarks[i][0],
                probability = rawPoseLandMarks[i][2],
                label = keypoints[i]!!
            )
            foundPoseLandmarks.add(i, poseLandmark)
        }
        val detectedPose = DetectedPose(foundPoseLandmarks, emptyList())

        val displayedImage = pipeline<BufferedImage>()
            .resize { outputWidth = 300; outputHeight = 300 }
            .apply(image)
        showFrame("Detection result for $file", createDetectedPosePanel(displayedImage, detectedPose))
    }
}

/** */
fun main(): Unit = poseDetectionMoveNet()
