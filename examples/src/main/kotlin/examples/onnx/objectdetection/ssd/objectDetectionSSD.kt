/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.SSDObjectDetectionModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedObjects
import java.io.File

fun main() {
    val modelHub =
        ONNXModelHub(commonModelDirectory = File("cache/pretrainedModels"), modelType = ONNXModels.ObjectDetection.SSD)
    val model = modelHub.loadModel() as SSDObjectDetectionModel

    model.use { detectionModel ->
        println(detectionModel)

        val imageFile = getFileFromResource("datasets/vgg/image9.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile, topK = 40)

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
        transformImage {
            load {
                pathToData = imageFile
                imageShape = ImageShape(224, 224, 3)
                colorMode = ColorOrder.BGR
            }
            resize {
                outputWidth = 1200
                outputHeight = 1200
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    drawDetectedObjects(rawImage, ImageShape(1200, 1200, 3), detectedObjects)
}

