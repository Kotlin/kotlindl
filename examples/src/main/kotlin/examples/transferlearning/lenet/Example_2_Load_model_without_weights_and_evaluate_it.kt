/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.lenet

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

/**
 * This examples demonstrates the weird inference case:
 * - Weights are not loaded, but initialized via initialized defined in configuration, configuration is loaded from .json file.
 * - Model is evaluated after loading to obtain accuracy value.
 * - No additional training.
 * - No new layers are added.
 *
 * NOTE: Model and weights are resources in `examples` module.
 */
fun loadModelWithoutWeightsInitAndEvaluate() {
    val (_, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.init()
        it.logSummary()

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy training $accuracy")
    }
}

/** */
fun main(): Unit = loadModelWithoutWeightsInitAndEvaluate()




