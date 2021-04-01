/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning


import examples.transferlearning.modelzoo.vgg16.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelLoader
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5Labels
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.imagePreprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocessingPipeline
import java.io.File

/**
 * This examples demonstrates the inference concept on ResNet'101 model:
 *
 * Weights are loaded from .h5 file, configuration is loaded from .json file.
 *
 * Model predicts on a few images located in resources.
 *
 * No additional training.
 *
 * No new layers are added.
 */
fun main() {
    val modelLoader =
        ModelLoader(commonModelDirectory = File("savedmodels/keras_models"), modelType = ModelType.ResNet_101)
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
            val preprocessing: Preprocessing = preprocessingPipeline {
                imagePreprocessing {
                    load {
                        pathToData = getFileFromResource("datasets/vgg/image$i.jpg")
                        imageShape = ImageShape(224, 224, 3)
                        colorMode = ColorOrder.BGR
                    }
                }
            }

            val inputData = modelLoader.preprocessInput(preprocessing().first, model.inputDimensions)
            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}


