/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.modelhub.vgg16

import examples.transferlearning.getFileFromResource
import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
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
 * This examples demonstrates the inference concept on VGG'16 model and weights loading from outdated or custom weights' schema in .h5 file:
 *
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model predicts on a few images located in resources.
 * - Special preprocessing (used in VGG'16 during training on ImageNet dataset) is applied to each image before prediction.
 * - No additional training.
 * - No new layers are added.
 *
 * NOTE: Also recursivePrintGroupInHDF5File() is helpful to discover hidden schema and paths.
 *
 * @see <a href="https://arxiv.org/abs/1409.1556">
 *     Very Deep Convolutional Networks for Large-Scale Image Recognition (ICLR 2015).</a>
 * @see <a href="https://keras.io/api/applications/vgg/#vgg16-function">
 *    Detailed description of VGG'16 model and an approach to build it in Keras.</a>
 */
fun main() {
    val jsonConfigFile = getVGG16JSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = Imagenet.V1k.labels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        it.logSummary()

        val hdfFile = getVGG16WeightsFile()
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        val kernelDataPathTemplate = "/%s/%s_W_1:0"
        val biasDataPathTemplate = "/%s/%s_b_1:0"
        it.loadWeightsByPathTemplates(hdfFile, kernelDataPathTemplate, biasDataPathTemplate)

        val fileLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(InputType.CAFFE.preprocessing())
            .fileLoader()

        for (i in 1..8) {
            val inputData = fileLoader.load(getFileFromResource("datasets/vgg/image$i.jpg")).first
            val res = it.predict(inputData, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }

    val model2 = Sequential.loadModelConfiguration(jsonConfigFile)
    model2.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        it.logSummary()

        val hdfFile = getVGG16WeightsFile()
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        val weightPaths = listOf(
            LayerConvOrDensePaths(
                "block1_conv1",
                "/block1_conv1/block1_conv1_W_1:0",
                "/block1_conv1/block1_conv1_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block1_conv2",
                "/block1_conv2/block1_conv2_W_1:0",
                "/block1_conv2/block1_conv2_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block2_conv1",
                "/block2_conv1/block2_conv1_W_1:0",
                "/block2_conv1/block2_conv1_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block2_conv2",
                "/block2_conv2/block2_conv2_W_1:0",
                "/block2_conv2/block2_conv2_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block3_conv1",
                "/block3_conv1/block3_conv1_W_1:0",
                "/block3_conv1/block3_conv1_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block3_conv2",
                "/block3_conv2/block3_conv2_W_1:0",
                "/block3_conv2/block3_conv2_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block3_conv3",
                "/block3_conv3/block3_conv3_W_1:0",
                "/block3_conv3/block3_conv3_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block4_conv1",
                "/block4_conv1/block4_conv1_W_1:0",
                "/block4_conv1/block4_conv1_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block4_conv2",
                "/block4_conv2/block4_conv2_W_1:0",
                "/block4_conv2/block4_conv2_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block4_conv3",
                "/block4_conv3/block4_conv3_W_1:0",
                "/block4_conv3/block4_conv3_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block5_con2v1",
                "/block5_conv1/block5_conv21_W_1:0",
                "/block5_conv1/block5_conv1_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block5_conv2",
                "/block5_conv2/block5_conv2_W_1:0",
                "/block5_conv2/block5_conv2_b_1:0"
            ),
            LayerConvOrDensePaths(
                "block5_conv3",
                "/block5_conv3/block5_conv3_W_1:0",
                "/block5_conv3/block5_conv3_b_1:0"
            ),
            LayerConvOrDensePaths("fc1", "/fc1/fc1_W_1:0", "/fc1/fc1_b_1:0"),
            LayerConvOrDensePaths("fc2", "/fc2/fc2_W_1:0", "/fc2/fc2_b_1:0"),
            LayerConvOrDensePaths("predictions", "/predictions/predictions_W_1:0", "/predictions/predictions_b_1:0"),
        )
        it.loadWeightsByPaths(hdfFile, weightPaths)

        val inputStreamLoader = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.BGR }
            .toFloatArray { }
            .call(InputType.CAFFE.preprocessing())
            .inputStreamLoader()

        for (i in 1..8) {
            val inputStream = OnHeapDataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val inputData = inputStreamLoader.load(inputStream).first
            val res = it.predict(inputData, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = it.predictTop5Labels(inputData, imageNetClassLabels)

            println(top5.toString())
        }
    }

    val model3 = Sequential.loadModelConfiguration(jsonConfigFile)
    model3.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        it.logSummary()

        val hdfFile = getVGG16WeightsFile()
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        it.loadWeights(hdfFile) // await exception
    }
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
private fun getVGG16JSONConfigFile(): File {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg16JSONModelPathForOldWeightSchema = properties["vgg16JSONModelPathForOldWeightSchema"] as String

    return File(vgg16JSONModelPathForOldWeightSchema)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. with old weights' schema in h5 file */
private fun getVGG16WeightsFile(): HdfFile {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val vgg19h5WeightsPathForOldWeightSchema = properties["vgg19h5WeightsPathForOldWeightSchema"] as String

    return HdfFile(File(vgg19h5WeightsPathForOldWeightSchema))
}


