/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.singlepose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedPose
import java.io.File

/**
 * This examples demonstrates the inference concept on MoveNetSinglePoseLighting model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun poseDetectionMoveNetLightAPI() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.PoseEstimation.MoveNetSinglePoseLighting.pretrainedModel(modelHub)

    model.use { poseDetectionModel ->
        val preprocessing = preprocessing()
        for (i in 1..3) {
            val imageFile = getFileFromResource("datasets/poses/single/$i.jpg")
            val detectedPose = poseDetectionModel.detectPose(imageFile = imageFile)

            detectedPose.poseLandmarks.forEach {
                println("Found ${it.poseLandmarkLabel} with probability ${it.probability}")
            }

            val (rawImage, shape) = preprocessing(imageFile)
            drawDetectedPose(rawImage, shape, detectedPose)
        }
    }
}

private fun preprocessing(): Preprocessing {
    return preprocess {
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
}

/** */
fun main(): Unit = poseDetectionMoveNetLightAPI()

