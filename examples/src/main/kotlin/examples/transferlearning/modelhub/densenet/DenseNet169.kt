/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.densenet


import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.LayerBatchNormPaths
import org.jetbrains.kotlinx.dl.api.inference.keras.LayerConvOrDensePaths
import org.jetbrains.kotlinx.dl.api.inference.keras.MissedWeightsStrategy
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsByPaths
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModelHub
import org.jetbrains.kotlinx.dl.api.inference.loaders.TFModels
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTop5Labels
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File

/**
 * This examples demonstrates the inference concept on DenseNet169 model:
 * - Model configuration, model weights and labels are obtained from [TFModelHub].
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in DenseNet169 during training on ImageNet dataset) is applied to each image before prediction.
 *
 * NOTE: Input resolution is 224*224
 */
fun denseNet169Prediction() {
    val modelHub = TFModelHub(cacheDirectory = File("cache/pretrainedModels"))
    val modelType = TFModels.CV.DenseNet169()
    val model = modelHub.loadModel(modelType)

    val imageNetClassLabels = modelHub.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = modelHub.loadWeights(modelType)

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
        it.loadWeightsByPaths(hdfFile, weightPaths, missedWeights = MissedWeightsStrategy.LOAD_CUSTOM_PATH)

        val fileDataLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(modelType.preprocessor)
            .fileLoader()

        for (i in 1..8) {
            val inputData = fileDataLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** */
fun main(): Unit = denseNet169Prediction()
