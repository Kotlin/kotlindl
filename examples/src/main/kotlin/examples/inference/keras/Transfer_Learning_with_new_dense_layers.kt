/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras

import LeNetClassic.SEED
import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.handlers.*
import java.io.File

/**
 * Conv2D layers' weights are loaded from ImageNet, Dense weights are initialized by loaded initializers.
 */
fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        NUMBER_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    val pathToWeights = "models/mnist/lenet/lenet_weights_only.h5"
    val realPathToWeights = Dataset::class.java.classLoader.getResource(pathToWeights).path.toString()
    val file = File(realPathToWeights)
    val hdfFile = HdfFile(file)

    val pathToConfig = "models/mnist/lenet/lenetMdl.json"
    val realPathToConfig = Dataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)

    val (input, otherLayers) = Sequential.loadModelLayersFromConfiguration(jsonConfigFile)

    val layers = mutableListOf<Layer>()
    for (layer in otherLayers) {
        if (layer::class == Conv2D::class || layer::class == MaxPool2D::class) {
            layer.isTrainable = false
            layers.add(layer)
        }
    }

    layers.add(Flatten("new_flatten"))
    layers.add(
        Dense(
            name = "new_dense_1",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 256,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_2",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 128,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_3",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 64,
            activation = Activations.Relu
        )
    )
    layers.add(
        Dense(
            name = "new_dense_4",
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            outputSize = 10,
            activation = Activations.Linear
        )
    )
    val model = Sequential.of(input, layers)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()

        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100,
            verbose = false
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}




