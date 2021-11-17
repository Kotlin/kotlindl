/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */
package examples.transferlearning.modelhub.resnet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import java.io.File

/**
 * This examples demonstrates the inference concept on ResNet'50 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - No additional training.
 * - No new layers are added.
 * - Special preprocessing (used in ResNet'50 during training on ImageNet dataset) is applied to images before prediction.
 */
fun resnet50easyPrediction2() {
    val modelHub =
        TFModelHub(cacheDirectory = File("cache/pretrainedModels"))

    val model = modelHub[TFModels.CV.ResNet50()]

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
fun main(): Unit = resnet50easyPrediction2()
