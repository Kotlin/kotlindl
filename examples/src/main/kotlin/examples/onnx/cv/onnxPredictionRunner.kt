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
import java.io.File

fun runONNXImageRecognitionPrediction(modelType: ONNXModels.CV) {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val imageNetClassLabels = Imagenet.V1k.labels()

    modelHub.loadModel(modelType).use { model ->
        println(model)

        val fileDataLoader = modelType.createPreprocessing(model).fileLoader()

        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = model.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = model.predictTopNLabels(inputData, imageNetClassLabels)
            println(top5.toString())
        }
    }
}

