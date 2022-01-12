/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.efficientdet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.objectdetection.DetectedObject
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.visualization.swing.drawDetectedObjects
import java.io.File

fun main() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.EfficientDetD2.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)

        val imageFile = getFileFromResource("datasets/detection/image3.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile)

        detectedObjects.forEach {
            println("Found ${it.classLabel} with probability ${it.probability}")
        }

        visualise(imageFile, detectedObjects, detectionModel.inputShape)
    }
}

internal fun visualise(
    imageFile: File,
    detectedObjects: List<DetectedObject>,
    inputShape: LongArray
) {
    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageFile
            imageShape = ImageShape(null, null, 3)
        }
        transformImage {
            resize {
                outputWidth = inputShape[1].toInt()
                outputHeight = inputShape[2].toInt()
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

    drawDetectedObjects(rawImage, ImageShape(inputShape[1], inputShape[2], 3), detectedObjects)
}


