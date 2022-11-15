/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */
package examples.onnx.cv.efficicentnet.lightAPI

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.summary.printSummary
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.ImageRecognitionModel
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModelHub
import org.jetbrains.kotlinx.dl.onnx.inference.ONNXModels
import java.io.File

/**
 * This examples demonstrates the light-weight inference API with [ImageRecognitionModel] on EfficientNetB7 model:
 * - Model is obtained from [ONNXModelHub].
 * - Model predicts on a few images located in resources.
 */
fun efficientNetB7LightAPIPrediction() {
    val modelHub =
        ONNXModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val model = ONNXModels.CV.EfficientNetB7().pretrainedModel(modelHub)
    model.printSummary()

    model.use {
        for (i in 1..8) {
            val imageFile = getFileFromResource("datasets/vgg/image$i.jpg")

            val recognizedObject = it.predictObject(imageFile = imageFile)
            println(recognizedObject)

            val top5 = it.predictTopKObjects(imageFile = imageFile, topK = 5)
            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = efficientNetB7LightAPIPrediction()
