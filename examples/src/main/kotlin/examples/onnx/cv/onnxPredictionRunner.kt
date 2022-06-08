/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.util.loadImageNetClassLabels
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.api.inference.onnx.OnnxInferenceModel
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import java.io.File

fun runONNXImageRecognitionPrediction(
    modelType: ONNXModels.CV<out OnnxInferenceModel>,
    resizeTo: Pair<Int, Int> = Pair(224, 224)
) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = loadImageNetClassLabels()

    model.use {
        println(it)

        val preprocessing: Preprocessing = examples.transferlearning.preprocessing(resizeTo)
        for (i in 1..8) {
            val image = preprocessing(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val inputData = modelType.preprocessInput(image, model.inputDimensions)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTopNLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

