/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.posedetection.multipose

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel.Companion.preprocessInput
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.rescale
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.visualization.swing.drawRawMultiPoseLandMarks
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun multiPoseDetectionMoveNet() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseDetection.MoveNetMultiPoseLighting
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        val imageFile = getFileFromResource("datasets/poses/multi/2.jpg")
        val preprocessing = pipeline<BufferedImage>()
            .resize {
                    outputHeight = 256
                    outputWidth = 256
                }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray {  }

        val inputData = modelType.preprocessInput(imageFile, preprocessing)
        val yhat = it.predictRaw(inputData)
        println(yhat.values.toTypedArray().contentDeepToString())
        visualisePoseLandmarks(imageFile, (yhat["output_0"] as Array<Array<FloatArray>>)[0])
    }
}

private fun visualisePoseLandmarks(
    imageFile: File,
    poseLandmarks: Array<FloatArray>
) {
    val preprocessing = pipeline<BufferedImage>()
        .resize {
                outputHeight = 256
                outputWidth = 256
            }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray {  }
        .rescale {
            scalingCoefficient = 255f
        }

    val (rawImage, shape) = preprocessing.dataLoader().load(imageFile)
    drawRawMultiPoseLandMarks(rawImage, shape.toImageShape(), poseLandmarks)
}

/** */
fun main(): Unit = multiPoseDetectionMoveNet()

