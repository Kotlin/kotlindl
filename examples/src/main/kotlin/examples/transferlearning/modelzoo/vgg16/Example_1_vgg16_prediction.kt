/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelzoo.vgg16


import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelLoader
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.ModelType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5Labels
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File


/**
 * This examples demonstrates the inference concept on VGG'16 model:
 *
 * Weights are loaded from .h5 file, configuration is loaded from .json file.
 *
 * Model predicts on a few images located in resources.
 *
 * No additional training.
 *
 * No new layers are added.
 *
 * NOTE: The specific image preprocessing is not implemented yet (see Keras for more details).
 *
 * @see <a href="https://drive.google.com/drive/folders/1283PvmF8TykZ70NVbLr1-gW0I6Y2rQ6Q">
 *     VGG'16 weights and model could be loaded here.</a>
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
 *    Detailed description of VGG'16 model and an approach to build it in Keras.</a>
 */
fun main() {
    // TODO: modelLoader rename
    val modelLoader = ModelLoader(commonModelDirectory = File("savedmodels/keras_models"), modelType = ModelType.VGG_16)
    val model = modelLoader.loadModel() as Sequential

    val imageNetClassLabels = modelLoader.loadClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        println(it.kGraph)

        it.summary()

        val hdfFile = modelLoader.loadWeights()

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputStream = object {}.javaClass.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val inputData = modelLoader.preprocessInput(floatArray, model.inputDimensions)

            val res = it.predict(inputData, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}






