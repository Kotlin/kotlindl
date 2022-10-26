/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssdmobile

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.visualization.swing.createDetectedObjectsPanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the light-weight inference API with [SSDObjectDetectionModel] on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts rectangles for the detected objects on a few images located in resources.
 * - The detected rectangles related to the objects are drawn on the images used for prediction.
 */
fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(modelHub)
    model.printSummary()

    model.use { detectionModel ->
        println(detectionModel)

        val file = getFileFromResource("datasets/vgg/image9.jpg")
        val image = ImageConverter.toBufferedImage(file)
        val detectedObjects = detectionModel.detectObjects(image, topK = 50)

        detectedObjects.forEach {
            println("Found ${it.label} with probability ${it.probability}")
        }

        val displayedImage = pipeline<BufferedImage>()
            .resize { outputWidth = 1200; outputHeight = ((1200f / image.width) * image.height).toInt() }
            .apply(image)
        showFrame("Detection result for ${file.name}", createDetectedObjectsPanel(displayedImage, detectedObjects))
    }
}