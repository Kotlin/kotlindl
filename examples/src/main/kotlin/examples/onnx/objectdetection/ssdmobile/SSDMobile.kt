/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssdmobile

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * This examples demonstrates the inference concept on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun ssdMobile() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.SSDMobileNetV1
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)

        val preprocessing: Preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = it.inputShape[1].toInt()
                    outputWidth = it.inputShape[2].toInt()
                }
                convert { colorMode = ColorMode.BGR }
            }
        }
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

/** */
fun main(): Unit = ssdMobile()

