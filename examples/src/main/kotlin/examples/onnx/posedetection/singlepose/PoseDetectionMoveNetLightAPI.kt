/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.singlepose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedPosePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.FlowLayout
import java.awt.image.BufferedImage
import java.io.File
import javax.swing.JPanel

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to each image before prediction.
 */
fun poseDetectionMoveNetLightAPI() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.PoseDetection.MoveNetSinglePoseLighting.pretrainedModel(modelHub)
    model.printSummary()

    model.use { poseDetectionModel ->
        val result = mutableMapOf<BufferedImage, DetectedPose>()
        for (i in 1..3) {
            val file = getFileFromResource("datasets/poses/single/$i.jpg")
            val image = ImageConverter.toBufferedImage(file)
            val detectedPose = poseDetectionModel.detectPose(image)

            detectedPose.landmarks.forEach {
                println("Found ${it.label} with probability ${it.probability}")
            }

            detectedPose.edges.forEach {
                println("The ${it.label} starts at ${it.start.label} and ends with ${it.end.label}")
            }

            result[image] = detectedPose
        }

        val panel = JPanel(FlowLayout(FlowLayout.CENTER, 0, 0))
        val height = 300
        for ((image, detectedPose) in result) {
            val displayedImage = pipeline<BufferedImage>()
                .resize { outputWidth = (height * image.width) / image.height; outputHeight = height }
                .apply(image)
            panel.add(createDetectedPosePanel(displayedImage, detectedPose))
        }
        showFrame("Detection results", panel)
    }
}

/** */
fun main(): Unit = poseDetectionMoveNetLightAPI()
