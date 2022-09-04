/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.singlepose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedPose
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun poseDetectionMoveNetLightAPI() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

    model.use { poseDetectionModel ->
        for (i in 1..3) {
            val image = ImageConverter.toBufferedImage(getFileFromResource("datasets/poses/single/$i.jpg"))
            val detectedPose = poseDetectionModel.detectPose(image)

            detectedPose.poseLandmarks.forEach {
                println("Found ${it.poseLandmarkLabel} with probability ${it.probability}")
            }

            detectedPose.edges.forEach {
                println("The ${it.poseEdgeLabel} starts at ${it.start.poseLandmarkLabel} and ends with ${it.end.poseLandmarkLabel}")
            }

            val displayedImage = pipeline<BufferedImage>()
                .resize { outputWidth = 300; outputHeight = 300 }
                .apply(image)
            drawDetectedPose(displayedImage, detectedPose)
        }
    }
}

/** */
fun main(): Unit = poseDetectionMoveNetLightAPI()
