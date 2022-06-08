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

/** Just loading ToyResNet trained in Keras. */
fun main() {
    val (_, test) = fashionMnist()

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

        println(it.kGraph)

        val accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")
    }
}


