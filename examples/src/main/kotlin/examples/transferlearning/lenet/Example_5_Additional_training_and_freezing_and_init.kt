/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.lenet

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

/**
 * This examples demonstrates the transfer learning concept:
 * - Weights are loaded from .h5 file for a pre-filtered list of layers (Conv2D only), configuration is loaded from .json file.
 * - Conv2D layer are added to the new Neural Network, its weights are frozen, Dense layers are added too and its weights are initialized via defined initializers.
 * - No new layers are added.
 *
 * NOTE: Model and weights are resources in `examples` module.
 */
fun additionalTrainingAndPartialFreezingAndPartialInitialization() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {
        // Freeze conv2d layers, keep dense layers trainable
        val layerList = it.layers.filterIsInstance<Conv2D>()
        layerList.forEach(Layer::freeze)

        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile, layerList)

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
fun main(): Unit = additionalTrainingAndPartialFreezingAndPartialInitialization()





