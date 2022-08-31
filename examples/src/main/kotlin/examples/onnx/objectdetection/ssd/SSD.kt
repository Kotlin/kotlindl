/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.fileLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to images before prediction.
 */
fun ssd() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.SSD
    val model = modelHub.loadModel(modelType)

    model.use {
        println(it)
        val preprocessing = pipeline<BufferedImage>()
            .resize {
                outputHeight = 1200
                outputWidth = 1200
            }
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 1..6) {
            val inputData = preprocessing.load(getFileFromResource("datasets/detection/image$i.jpg")).first

            val start = System.currentTimeMillis()
            val yhat = it.predictRaw(inputData)
            val end = System.currentTimeMillis()
            println("Prediction took ${end - start} ms")
            println(yhat.values.toTypedArray().contentDeepToString())
        }
    }
}

/** */
fun main(): Unit = ssd()
