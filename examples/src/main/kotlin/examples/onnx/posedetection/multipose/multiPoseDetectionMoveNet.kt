/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.multipose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.posedetection.DetectedPose
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.api.inference.posedetection.PoseLandmark
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.get2DFloatArray
import org.jetbrains.kotlinx.dl.visualization.swing.createMultipleDetectedPosesPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to each image before prediction.
 */
fun multiPoseDetectionMoveNet() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseDetection.MoveNetMultiPoseLighting
    val model = modelHub.loadModel(modelType)
    model.printSummary()

    model.use {
        println(it)

        val imageFile = getFileFromResource("datasets/poses/multi/2.jpg")
        val inputImage = ImageConverter.toBufferedImage(imageFile)

        val preprocessor = pipeline<BufferedImage>()
            .resize {
                outputHeight = 256
                outputWidth = 256
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)

        val inputData = preprocessor.apply(inputImage)
        val rawPoseLandmarks = it.predictRaw(inputData) { result ->
            result.get2DFloatArray("output_0")
        }
        println(rawPoseLandmarks.contentDeepToString())

        val poses = rawPoseLandmarks.mapNotNull { floats ->
            val probability = floats[55]
            if (probability < 0.05) return@mapNotNull null

            val foundPoseLandmarks = mutableListOf<PoseLandmark>()

            for (keyPointIdx in 0..16) {
                val poseLandmark = PoseLandmark(
                    x = floats[3 * keyPointIdx + 1],
                    y = floats[3 * keyPointIdx],
                    probability = floats[3 * keyPointIdx + 2],
                    label = ""
                )
                foundPoseLandmarks.add(poseLandmark)
            }

            // [ymin, xmin, ymax, xmax, score]
            val detectedObject = DetectedObject(
                xMin = floats[52],
                xMax = floats[54],
                yMin = floats[51],
                yMax = floats[53],
                probability = probability
            )
            val detectedPose = DetectedPose(foundPoseLandmarks, emptyList())

            detectedObject to detectedPose
        }

        val multiPoseDetectionResult = MultiPoseDetectionResult(poses)
        showFrame(
            "Detection result for ${imageFile.name}",
            createMultipleDetectedPosesPanel(inputImage, multiPoseDetectionResult)
        )
    }
}

/** */
fun main(): Unit = multiPoseDetectionMoveNet()

