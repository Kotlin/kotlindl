/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.executionproviders

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CUDA
import org.jetbrains.kotlinx.dl.onnx.inference.inferUsing
import java.awt.image.BufferedImage
import java.io.File
import kotlin.system.measureTimeMillis

/**
 * This example compares the inference speed of different execution providers:
 * - [inferUsing] scope function is used for CUDA inference. That's why the underlying session should be closed manually.
 */
fun multiPoseCudaInference() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = ONNXModels.PoseDetection.MoveNetMultiPoseLighting
    // explicitly specify CPU execution provider during model loading
    val model = modelHub.loadModel(modelType, CPU())

    val inputData = prepareInputData(modelType)

    val cpuInferenceTime = cpuInference(model, inputData)
    println("Average inference time on CPU: $cpuInferenceTime ms")

    val cudaInferenceTime = cudaInference(model, inputData)
    println("Average inference time on CUDA: $cudaInferenceTime ms")

    model.close()
}

fun prepareInputData(modelType: ONNXModels.PoseDetection.MoveNetMultiPoseLighting): FloatData {
    val imageFile = getFileFromResource("datasets/poses/multi/2.jpg")
    val fileDataLoader = pipeline<BufferedImage>()
        .resize {
            outputHeight = 256
            outputWidth = 256
        }
        .convert { colorMode = ColorMode.RGB }
        .toFloatArray { }
        .call(modelType.preprocessor)
        .fileLoader()

    return fileDataLoader.load(imageFile)
}

fun cpuInference(model: OnnxInferenceModel, inputData: FloatData, n: Int = 10): Long {
    val totalPredictionTime = model.run {
        measureTimeMillis {
            repeat(n) { predictRaw(inputData) }
        }
    }

    return totalPredictionTime / n
}

fun cudaInference(model: OnnxInferenceModel, inputData: FloatData, n: Int = 10): Long {
    /**
     * First inference on GPU takes way longer than average due to model serialization
     * and GPU memory buffers initialization.
     * Making a dummy inference to 'warm up' GPU is recommended to avoid wrong inference time calculation.
     */
    model.inferUsing(CUDA(0)) {
        it.predictRaw(inputData)
    }

    val totalPredictionTime =
        measureTimeMillis {
            repeat(n) {
                model.inferUsing(CUDA(0)) {
                    it.predictRaw(inputData)
                }
            }
        }

    return totalPredictionTime / n
}

fun main(): Unit = multiPoseCudaInference()
