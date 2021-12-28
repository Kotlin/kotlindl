/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/**
 * So, let's update the weights of last layer of the pretrained model from Keras.
 *
 * All layers except last should be frozen.
 *
 * As a result the training will be fast.
 */
fun main() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFileToyResNet()
    val model = Functional.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getWeightsFileToyResNet()

        it.loadWeights(hdfFile)

        var accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")

        for (layer in it.layers) {
            layer.isTrainable = false
        }
        it.layers.last().isTrainable = true

        it.fit(dataset = train, epochs = 1, batchSize = 1000)

        accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy after: $accuracy")
    }
}


