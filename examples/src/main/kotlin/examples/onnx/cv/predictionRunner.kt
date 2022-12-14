/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTopNLabels
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels.CV.Companion.createPreprocessing
import org.jetbrains.kotlinx.dl.onnx.inference.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.onnx.inference.inferAndCloseUsing
import java.io.File

fun runImageRecognitionPrediction(
    modelType: ONNXModels.CV,
    executionProviders: List<ExecutionProvider> = emptyList()
): List<Pair<String, Float>> {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = Imagenet.V1k.labels()

    val inference: (OnnxInferenceModel) -> List<Pair<String, Float>> = {
        println(it)

        val fileDataLoader = modelType.createPreprocessing(model).fileLoader()

        val results = mutableListOf<Pair<String, Float>>()
        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            results.addAll(it.predictTopNLabels(inputData, imageNetClassLabels))
        }

        results
    }

    return if (executionProviders.isNotEmpty()) {
        model.inferAndCloseUsing(*executionProviders.toTypedArray()) { inference(it) }
    } else {
        model.use { inference(it) }
    }

}
