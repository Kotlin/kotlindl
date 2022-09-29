/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.onnx.cv.efficicentnet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.util.predictTopNLabels
import org.jetbrains.kotlinx.dl.api.inference.loaders.ONNXModelHub
import org.jetbrains.kotlinx.dl.api.inference.onnx.ONNXModels
import org.jetbrains.kotlinx.dl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessing.call
import org.jetbrains.kotlinx.dl.dataset.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.fileLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.toFloatArray
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on EfficientNet4Lite model:
 * - Model configuration, model weights and labels are obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in EfficientNet4Lite during training on ImageNet dataset) is applied to images before prediction.
 */
fun efficientNet4LitePrediction() {
    val modelHub = ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val modelType = ONNXModels.CV.EfficientNet4Lite()
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = Imagenet.V1k.labels()

    model.use {
        println(it)

        val fileDataLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray {  }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTopNLabels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = efficientNet4LitePrediction()
