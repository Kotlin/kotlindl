/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.resnet

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5ImageNetLabels
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * This examples demonstrates the inference concept on ResNet'50 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - No additional training.
 * - No new layers are added.
 * - Special preprocessing (used in ResNet'50  during training on ImageNet dataset) is applied to images before prediction.
 * - Model copied and used for prediction.
 */
fun resnet50copyModelPrediction() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.ResNet50()
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = modelHub.loadClassLabels()

    var copiedModel: Functional

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = modelHub.loadWeights(modelType)

        it.loadWeights(hdfFile)

        copiedModel = it.copy(copyWeights = true)

        val preprocessing: Preprocessing = preprocess {
            transformImage { convert { colorMode = ColorMode.BGR } }
        }
        for (i in 1..8) {
            val image = preprocessing(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val inputData = modelType.preprocessInput(image, model.inputDimensions)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5ImageNetLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }

    copiedModel.use {
        val preprocessing: Preprocessing = preprocess {
            transformImage { convert { colorMode = ColorMode.BGR } }
        }
        for (i in 1..8) {
            val image = preprocessing(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val inputData = modelType.preprocessInput(image, model.inputDimensions)

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5ImageNetLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = resnet50copyModelPrediction()
