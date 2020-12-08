/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras.transferlearning

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

fun main() {
    val jsonConfigFilePath = "C:\\zaleslaw\\home\\models\\vgg\\modelConfig.json"
    val jsonConfigFile = File(jsonConfigFilePath)
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    val imageNetClassLabels = prepareHumanReadableClassLabels()

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.ACCURACY
        )
        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        val kernelDataPathTemplate = "/%s/%s_W_1:0"
        val biasDataPathTemplate = "/%s/%s_b_1:0"
        it.loadWeightsByPathTemplates(hdfFile, kernelDataPathTemplate, biasDataPathTemplate)

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val res = it.predict(floatArray, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, floatArray, imageNetClassLabels)

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
        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        val weightPaths = listOf(
            LayerKernelAndBiasPaths(
                "block1_conv1",
                "/block1_conv1/block1_conv1_W_1:0",
                "/block1_conv1/block1_conv1_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block1_conv2",
                "/block1_conv2/block1_conv2_W_1:0",
                "/block1_conv2/block1_conv2_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block2_conv1",
                "/block2_conv1/block2_conv1_W_1:0",
                "/block2_conv1/block2_conv1_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block2_conv2",
                "/block2_conv2/block2_conv2_W_1:0",
                "/block2_conv2/block2_conv2_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block3_conv1",
                "/block3_conv1/block3_conv1_W_1:0",
                "/block3_conv1/block3_conv1_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block3_conv2",
                "/block3_conv2/block3_conv2_W_1:0",
                "/block3_conv2/block3_conv2_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block3_conv3",
                "/block3_conv3/block3_conv3_W_1:0",
                "/block3_conv3/block3_conv3_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block4_conv1",
                "/block4_conv1/block4_conv1_W_1:0",
                "/block4_conv1/block4_conv1_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block4_conv2",
                "/block4_conv2/block4_conv2_W_1:0",
                "/block4_conv2/block4_conv2_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block4_conv3",
                "/block4_conv3/block4_conv3_W_1:0",
                "/block4_conv3/block4_conv3_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block5_con2v1",
                "/block5_conv1/block5_conv21_W_1:0",
                "/block5_conv1/block5_conv1_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block5_conv2",
                "/block5_conv2/block5_conv2_W_1:0",
                "/block5_conv2/block5_conv2_b_1:0"
            ),
            LayerKernelAndBiasPaths(
                "block5_conv3",
                "/block5_conv3/block5_conv3_W_1:0",
                "/block5_conv3/block5_conv3_b_1:0"
            ),
            LayerKernelAndBiasPaths("fc1", "/fc1/fc1_W_1:0", "/fc1/fc1_b_1:0"),
            LayerKernelAndBiasPaths("fc2", "/fc2/fc2_W_1:0", "/fc2/fc2_b_1:0"),
            LayerKernelAndBiasPaths("predictions", "/predictions/predictions_W_1:0", "/predictions/predictions_b_1:0"),
        )
        it.loadWeightsByPaths(hdfFile, weightPaths)

        for (i in 1..8) {
            val inputStream = Dataset::class.java.classLoader.getResourceAsStream("datasets/vgg/image$i.jpg")
            val floatArray = ImageConverter.toRawFloatArray(inputStream)

            val res = it.predict(floatArray, "Activation_predictions")
            println("Predicted object for image$i.jpg is ${imageNetClassLabels[res]}")

            val top5 = predictTop5Labels(it, floatArray, imageNetClassLabels)

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
        it.summary()

        val pathToWeights = "C:\\zaleslaw\\home\\models\\vgg\\hdf\\vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        val file = File(pathToWeights)
        val hdfFile = HdfFile(file)
        recursivePrintGroupInHDF5File(hdfFile, hdfFile)

        it.loadWeights(hdfFile) // await exception
    }
}


