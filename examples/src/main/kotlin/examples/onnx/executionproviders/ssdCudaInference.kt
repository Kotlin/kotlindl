/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */
package examples.onnx.executionproviders

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProviders.ExecutionProvider.CUDA
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferAndCloseUsing
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
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
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = 1200
                    outputWidth = 1200
                }
                convert { colorMode = ColorMode.RGB }
            }
        }
        for (i in 1..6) {
            val inputData = modelType.preprocessInput(
                getFileFromResource("datasets/detection/image$i.jpg"),
                preprocessing
            )

            val start = System.currentTimeMillis()
            val yhat = it.predictRaw(inputData)
            val end = System.currentTimeMillis()
            println("Prediction took ${end - start} ms")
            println(yhat.values.toTypedArray().contentDeepToString())
        }
    }
}

fun main(): Unit = ssdCudaInference()
