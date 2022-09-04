/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.multipose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.visualization.swing.drawMultiPoseLandMarks
import java.io.File

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun multiPoseDetectionMoveNetLightAPI() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.PoseDetection.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

    model.use { poseDetectionModel ->
        for (i in 1..3) {
            val image = ImageConverter.toBufferedImage(getFileFromResource("datasets/poses/multi/$i.jpg"))
            val detectedPoses = poseDetectionModel.detectPoses(image = image, confidence = 0.0f)

            detectedPoses.multiplePoses.forEach { detectedPose ->
                println("Found ${detectedPose.first.classLabel} with probability ${detectedPose.first.probability}")
                detectedPose.second.poseLandmarks.forEach {
                    println("   Found ${it.poseLandmarkLabel} with probability ${it.probability}")
                }

                detectedPose.second.edges.forEach {
                    println("   The ${it.poseEdgeLabel} starts at ${it.start.poseLandmarkLabel} and ends with ${it.end.poseLandmarkLabel}")
                }
            }

            drawMultiPoseLandMarks(image, detectedPoses)
        }
    }
}

/** */
fun main(): Unit = multiPoseDetectionMoveNetLightAPI()
