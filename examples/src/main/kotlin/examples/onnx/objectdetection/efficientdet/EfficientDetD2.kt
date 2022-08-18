/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.efficientdet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.ImageRecognitionModel.Companion.preprocessInput
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File

fun main() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.EfficientDetD2
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        val preprocessing = pipeline<BufferedImage>()
            .resize {
                    outputHeight = it.inputShape[1].toInt()
                    outputWidth = it.inputShape[2].toInt()
                }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray {  }

        for (i in 1..6) {
            val inputData = modelType.preprocessInput(
                getFileFromResource("datasets/detection/image$i.jpg"),
                preprocessing
            )

            val yhat = it.predictRaw(inputData)
            println(yhat.values.toTypedArray().contentDeepToString())
        }
    }
}


