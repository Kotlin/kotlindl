/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.lenet

import LeNetClassic.SEED
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeightsForFrozenLayers
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

/**
 * This examples demonstrates the transfer learning concept:
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Conv2D layer are added to the new Neural Network, its weights are frozen.
 * - Flatten and new Dense layers are added and initialized via defined initializers.
 *
 * NOTE: Model and weights are resources in `examples` module.
 */
fun additionalTrainingAndNewTopDenseLayers() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val (input, otherLayers) = Sequential.loadModelLayersFromConfiguration(jsonConfigFile)

    val layers = mutableListOf<Layer>()
    layers.add(input)
    for (layer in otherLayers) {
        if (layer is Conv2D || layer is MaxPool2D) {
            layer.freeze()
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
    val model = Sequential.of(layers)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getWeightsFile()
        it.loadWeightsForFrozenLayers(hdfFile)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 5,
            trainBatchSize = 1000,
            validationBatchSize = 100
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}

/** */
fun main(): Unit = additionalTrainingAndNewTopDenseLayers()





