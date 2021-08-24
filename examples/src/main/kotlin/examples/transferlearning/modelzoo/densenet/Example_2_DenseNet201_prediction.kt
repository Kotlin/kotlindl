/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.densenet


import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.LayerBatchNormPaths
import org.jetbrains.kotlinx.dl.api.inference.keras.LayerConvOrDensePaths
import org.jetbrains.kotlinx.dl.api.inference.keras.MissedWeightsStrategy
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsByPaths
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5ImageNetLabels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File

/**
 * This examples demonstrates the inference concept on DenseNet201 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in DenseNet201 during training on ImageNet dataset) is applied to images before prediction.
 *
 * NOTE: Input resolution is 224*224
 */
fun denseNet201Prediction() {
    val modelHub =
        TFModelHub(commonModelDirectory = File("cache/pretrainedModels"), modelType = TFModels.CV.DenseNet201)
    val model = modelHub.loadModel() as Functional

    val imageNetClassLabels = modelHub.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = modelHub.loadWeights()

        val weightPaths = listOf(
            LayerConvOrDensePaths(
                "conv1_conv",
                "/conv1/conv/conv1/conv/kernel:0",
                "/conv1/conv/conv1/conv/bias:0"
            ),
            LayerBatchNormPaths(
                "conv1_bn",
                "/conv1/bn/conv1/bn/gamma:0",
                "/conv1/bn/conv1/bn/beta:0",
                "/conv1/bn/conv1/bn/moving_mean:0",
                "/conv1/bn/conv1/bn/moving_variance:0"
            )
        )
        it.loadWeightsByPaths(hdfFile, weightPaths, missedWeights = MissedWeightsStrategy.LOAD_NEW_FORMAT)

        for (i in 1..8) {
            val preprocessing: Preprocessing = preprocess {
                transformImage {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.RGB
                    }
                }
            }

            val inputData = modelHub.preprocessInput(preprocessing().first, model.inputDimensions)
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5ImageNetLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = denseNet201Prediction()
