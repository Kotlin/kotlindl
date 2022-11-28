/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg19


import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.preprocessing.inputStreamLoader
import org.jetbrains.kotlinx.dl.impl.dataset.Imagenet
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.InputType
import org.jetbrains.kotlinx.dl.impl.inference.imagerecognition.predictTop5Labels
import org.jetbrains.kotlinx.dl.impl.preprocessing.call
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.awt.image.BufferedImage
import java.io.File
import java.io.FileReader
import java.util.*

/**
 * This examples demonstrates the inference concept on VGG'19 model:
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in VGG'19 during training on ImageNet dataset) is applied to each image before prediction.
 * - No additional training.
 * - No new layers are added.
 *
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg19-function">
 *    Detailed description of VGG'19 model and an approach to build it in Keras.</a>
 */
fun main() {
    val jsonConfigFile = getVGG19JSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = Imagenet.V1k.labels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getVGG19WeightsFile()

        it.loadWeights(hdfFile)

        val inputStreamLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(InputType.CAFFE.preprocessing())
            .inputStreamLoader()

        for (i in 1..8) {
            val inputStream = OnHeapDataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val inputData = inputStreamLoader.load(inputStream).first

            val res = it.predict(inputData)
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
private fun getVGG19JSONConfigFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg19JSONModelPath = properties["vgg19JSONModelPath"] as String

    return File(vgg19JSONModelPath)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
private fun getVGG19WeightsFile(): HdfFile {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg19h5WeightsPath = properties["vgg19h5WeightsPath"] as String

    return HdfFile(File(vgg19h5WeightsPath))
}


