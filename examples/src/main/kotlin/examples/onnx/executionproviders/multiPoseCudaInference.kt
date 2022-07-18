/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.executionproviders

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProviders.ExecutionProvider.CUDA
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferUsing
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import java.io.File

/**
 * This example demonstrates how to infer MoveNetMultiPoseLighting model using [inferUsing] scope function:
 * - Model is obtained from [ONNXModelHub].
 * - Model detects poses on image located in resources using CUDA execution provider.
 * - Internal onnx session is not closed automatically after inference lambda is executed. The session is closed manually.
 */
fun multiPoseCudaInference() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseDetection.MoveNetMultiPoseLighting
    val model = modelHub.loadModel(modelType)

    model.inferUsing(CUDA(0)) {
        println(it)

        val imageFile = getFileFromResource("datasets/poses/multi/2.jpg")
        val preprocessing: Preprocessing = preprocess {
            transformImage {
                resize {
                    outputHeight = 256
                    outputWidth = 256
                }
                convert { colorMode = ColorMode.RGB }
            }
        }

        val inputData = modelType.preprocessInput(imageFile, preprocessing)

        val start = System.currentTimeMillis()
        val yhat = it.predictRaw(inputData)
        val end = System.currentTimeMillis()

        println("Prediction took ${end - start} ms")
        println(yhat.values.toTypedArray().contentDeepToString())
    }

    model.close()
}

fun main(): Unit = multiPoseCudaInference()
