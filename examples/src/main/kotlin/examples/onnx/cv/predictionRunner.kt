/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.imagerecognition.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.inferAndCloseUsing
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.fileLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File

fun runImageRecognitionPrediction(
    modelType: ONNXModels.CV<OnnxInferenceModel>,
    executionProviders: List<ExecutionProvider> = emptyList()
): List<Pair<String, Float>> {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels =
        loadImageNetClassLabels()

    val inference: (OnnxInferenceModel) -> List<Pair<String, Float>> = {
        println(it)

        val fileDataLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

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
        model.inferAndCloseUsing(executionProviders) { inference(it) }
    } else {
        model.use { inference(it) }
    }

}
