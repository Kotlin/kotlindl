/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg16

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.InputType
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.predictTop5ImageNetLabels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.prepareImageNetHumanReadableClassLabels
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.preprocessInput
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.convert
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import java.io.File
import java.io.FileReader
import java.util.*

/**
 * This examples demonstrates the inference concept on VGG'16 model and weights loading from KotlinDL txt format:
 * - Weights are loaded from txt files, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in VGG'16 during training on ImageNet dataset) is applied to images before prediction.
 * - No additional training.
 * - No new layers are added.
 *
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
 *    Detailed description of VGG'16 model and an approach to build it in Keras.</a>
 */
fun main() {
    val jsonConfigFile = getVGG16JSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = prepareImageNetHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        println(it.kGraph)

        it.logSummary()

        it.loadWeights(getVGG16WeightsFile())

        val preprocessing: Preprocessing = preprocess {
            transformImage { convert { colorMode = ColorMode.BGR } }
        }
        for (i in 1..8) {
            val image = preprocessing(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val inputData = preprocessInput(image, model.inputDimensions, inputType = InputType.CAFFE)
            val res = it.predict(inputData, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5ImageNetLabels(it, inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}


/** Returns JSON file with model configuration, saved from Keras 2.x. */
private fun getVGG16JSONConfigFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg16JSONModelPath = properties["vgg16JSONModelPath"] as String

    return File(vgg16JSONModelPath)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
private fun getVGG16WeightsFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg16h5TxtWeightsPath = properties["vgg16h5TxtWeightsPath"] as String

    return File(vgg16h5TxtWeightsPath)
}




