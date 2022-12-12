/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */
package examples.onnx.executionproviders

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CUDA
import org.jetbrains.kotlinx.dl.onnx.inference.inferAndCloseUsing
import java.awt.image.BufferedImage
import java.io.File

/**
 * This example demonstrates how to infer SSD model using [inferAndCloseUsing] scope function:
 * - Model is obtained from [ONNXModelHub].
 * - Model performs classification of a few images located in resources using CUDA execution provider.
 * - Internal onnx session is closed automatically after inference lambda is executed.
 */
fun ssdCudaInference() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.ObjectDetection.SSD
    val model = modelHub.loadModel(modelType)

    model.inferAndCloseUsing(CUDA()) {
        val preprocessing = pipeline<BufferedImage>()
            .resize {
                outputHeight = 1200
                outputWidth = 1200
            }
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 1..6) {
            val inputData = preprocessing.load(getFileFromResource("datasets/detection/image$i.jpg"))

            val start = System.currentTimeMillis()
            val yhat = it.predictRaw(inputData)
            val end = System.currentTimeMillis()
            println("Prediction took ${end - start} ms")
            println(yhat.values.toTypedArray().contentDeepToString())
        }
    }
}

fun main(): Unit = ssdCudaInference()
