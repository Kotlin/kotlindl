/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.multipose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.posedetection.MultiPoseDetectionResult
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
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
    val model = ONNXModels.PoseEstimation.MoveNetMultiPoseLighting.pretrainedModel(modelHub)

    model.use { poseDetectionModel ->
        for (i in 1..3) {
            val imageFile = getFileFromResource("datasets/poses/multi/$i.jpg")
            val detectedPoses = poseDetectionModel.detectPoses(imageFile = imageFile, confidence = 0.0f)

            detectedPoses.multiplePoses.forEach { detectedPose ->
                println("Found ${detectedPose.first.classLabel} with probability ${detectedPose.first.probability}")
                detectedPose.second.poseLandmarks.forEach {
                    println("   Found ${it.poseLandmarkLabel} with probability ${it.probability}")
                }
            }
            visualise(imageFile, detectedPoses)
        }
    }
}

private fun visualise(
    imageFile: File,
    multiPoseDetectionResult: MultiPoseDetectionResult
) {
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
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first
    drawMultiPoseLandMarks(rawImage, ImageShape(256, 256, 3), multiPoseDetectionResult)
}

/** */
fun main(): Unit = multiPoseDetectionMoveNetLightAPI()

