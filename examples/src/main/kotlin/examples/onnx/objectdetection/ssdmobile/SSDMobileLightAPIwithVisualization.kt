/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssdmobile

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.objectdetection.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedObjects
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
    val model = ONNXModels.ObjectDetection.SSDMobileNetV1.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)

        val imageFile = getFileFromResource("datasets/vgg/image9.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile, topK = 50)

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
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(null, null, 3)
        }
        transformImage {
            resize {
                outputWidth = 1000
                outputHeight = 1000
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

    drawDetectedObjects(rawImage, ImageShape(1000, 1000, 3), detectedObjects)
}

