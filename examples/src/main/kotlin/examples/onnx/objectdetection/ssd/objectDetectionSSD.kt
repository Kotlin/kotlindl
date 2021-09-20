/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import java.io.File

fun objectDetectionSSD() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = ONNXModels.ObjectDetection.SSD.pretrainedModel(modelHub)

    model.use { detectionModel ->
        println(detectionModel)

        val imageFile = getFileFromResource("datasets/vgg/image9.jpg")
        val detectedObjects =
            detectionModel.detectObjects(imageFile = imageFile, topK = 50)

        detectedObjects.forEach {
            println("Found ${it.classLabel} with probability ${it.probability}")
        }
    }
}

/** */
fun main(): Unit = objectDetectionSSD()


