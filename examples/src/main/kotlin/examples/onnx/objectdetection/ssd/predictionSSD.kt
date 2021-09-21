/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

fun predictionSSD() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.SSD
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        for (i in 0..8) {
            val preprocessing: Preprocessing = preprocess {
                load {
                    pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                    imageShape = ImageShape(224, 224, 3)
                    colorMode = ColorOrder.BGR
                }
                transformImage {
                    resize {
                        outputHeight = 1200
                        outputWidth = 1200
                    }
                }
            }

            val inputData = modelType.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)
            println(yhat.toTypedArray().contentDeepToString())
        }
    }
}

/** */
fun main(): Unit = predictionSSD()

