/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

fun main() {
    val modelHub =
        ONNXModelHub(commonModelDirectory = File("cache/pretrainedModels"), modelType = ONNXModels.ObjectDetection.SSD)
    val model = modelHub.loadModel() as OnnxInferenceModel

    model.use {
        println(it)

        for (i in 0..8) {
            val preprocessing: Preprocessing = preprocess {
                transformImage {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                    resize {
                        outputHeight = 1200
                        outputWidth = 1200
                    }
                }
            }

            val inputData = modelHub.preprocessInput(preprocessing)

            val yhat = it.predictRaw(inputData)
            println(yhat.toTypedArray().contentDeepToString())
        }
    }
}

