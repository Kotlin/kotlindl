/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.objectdetection.ssd

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OrtSessionResultConversions.getFloatArray
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on SSD model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing is applied to each image before prediction.
 */
fun ssd() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.SSD
    val model = modelHub.loadModel(modelType)
    model.printSummary()

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
            val inputData = preprocessing.load(getFileFromResource("datasets/detection/image$i.jpg"))

            val start = System.currentTimeMillis()
            val yhat = it.predictRaw(inputData) { output -> output.getFloatArray(0) }
            val end = System.currentTimeMillis()
            println("Prediction took ${end - start} ms")
            println(yhat.contentToString())
        }
    }
}

/** */
fun main(): Unit = ssd()
