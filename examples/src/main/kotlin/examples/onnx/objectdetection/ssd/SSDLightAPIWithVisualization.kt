/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.rescale
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedObjects
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the light-weight inference API with [SSDObjectDetectionModel] on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts rectangles for the detected objects on a few images located in resources.
 * - The detected rectangles related to the objects are drawn on the images used for prediction.
 */
fun main() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.SSD.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)

        val imageFile = getFileFromResource("datasets/detection/image2.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile, topK = 20)

        detectedObjects.forEach {
            println("Found ${it.classLabel} with probability ${it.probability}")
        }

        visualise(imageFile, detectedObjects)
    }
}

private fun visualise(
    imageFile: File,
    detectedObjects: List<DetectedObject>
) {
    val preprocessing = pipeline<BufferedImage>()
        .resize {
            outputWidth = 1200
            outputHeight = 1200
        }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .rescale {
            scalingCoefficient = 255f
        }

    val (rawImage, shape) = preprocessing.dataLoader().load(imageFile)

    drawDetectedObjects(rawImage, shape.toImageShape(), detectedObjects)
}
