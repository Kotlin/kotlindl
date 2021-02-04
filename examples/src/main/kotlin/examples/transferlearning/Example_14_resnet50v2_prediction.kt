/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning


import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File
import java.io.FileReader
import java.util.*

/**
 * This examples demonstrates the inference concept on VGG'19 model:
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
 * @see <a href="https://drive.google.com/drive/folders/1P1BlCNXdeXo_9u6mxYnm-N_gbOn_VhUA">
 *     VGG'19 weights and model could be loaded here.</a>
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg19-function">
 *    Detailed description of VGG'19 model and an approach to build it in Keras.</a>
 */
fun main() {
    val jsonConfigFile = getResNet50v2JSONConfigFile()
    val model = Functional.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = prepareHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.summary()

        val hdfFile = getResNet50v2WeightsFile()

        it.loadWeights(hdfFile)

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val res = it.predict(floatArray)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, floatArray, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
private fun getResNet50v2JSONConfigFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val resnet50v2JSONModelPath = properties["resnet50v2JSONModelPath"] as String

    return File(resnet50v2JSONModelPath)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
private fun getResNet50v2WeightsFile(): HdfFile {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val resnet50v2h5WeightsPath = properties["resnet50v2h5WeightsPath"] as String

    return HdfFile(File(resnet50v2h5WeightsPath))
}


