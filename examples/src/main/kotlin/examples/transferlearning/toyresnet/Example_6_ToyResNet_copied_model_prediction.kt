/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

/** Just loading ToyResNet trained in Keras, making a copy and using for prediction. */
fun main() {
    val (_, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFileToyResNet()
    val model = Functional.loadModelConfiguration(jsonConfigFile)
    var copiedModel: Functional
    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getWeightsFileToyResNet()

        it.loadWeights(hdfFile)
        copiedModel = it.copy(copyWeights = true)

        val accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")
    }

    copiedModel.use {
        copiedModel.logSummary()
        val accuracy = copiedModel.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")
    }
}



