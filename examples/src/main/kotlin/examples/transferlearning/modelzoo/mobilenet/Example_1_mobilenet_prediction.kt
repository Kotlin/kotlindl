/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.mobilenet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelLoader
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5Labels
import org.jetbrains.kotlinx.dl.datasets.OnHeapDataset
import org.jetbrains.kotlinx.dl.datasets.image.ColorOrder
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

/**
 * This examples demonstrates the inference concept on MobileNet model:
 *
 * Weights are loaded from .h5 file, configuration is loaded from .json file.
 *
 * Model predicts on a few images located in resources.
 */
fun main() {
    val modelLoader =
        ModelLoader(commonModelDirectory = File("savedmodels/keras_models"), modelType = ModelType.MobileNet)
    val model = modelLoader.loadModel() as Functional

    val imageNetClassLabels = modelLoader.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = modelLoader.loadWeights()

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputStream = OnHeapDataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream, ColorOrder.RGB)

            val inputData = modelLoader.preprocessInput(floatArray, model.inputDimensions)
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}
