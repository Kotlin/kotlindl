/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.experimental.batchnorm

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.unfreeze
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import java.io.File

/**
 * This examples demonstrates the inference concept:
 *
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model is evaluated after loading to obtain accuracy value.
 * - No additional training.
 * - No new layers are added.
 *
 * NOTE: Model and weights are resources in api module.
 *
 * We demonstrate the workflow on the FashionMnist classification dataset.
 */
fun main() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {

        it.freeze()
        it.layers.last().unfreeze()

        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()
        println(it.kGraph)
        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile)

        var accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy before training $accuracy")

        it.fit(dataset = train, epochs = 1, batchSize = 100)
        println(it.kGraph)
        accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]
        println(it.kGraph)
        println("Accuracy after training $accuracy")
    }
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
fun getJSONConfigFile(): File {
    val pathToConfig = "models/batchnorm/modelConfig.json"
    val realPathToConfig = OnHeapDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    return File(realPathToConfig)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
fun getWeightsFile(): HdfFile {
    val pathToWeights = "models/batchnorm/weights.h5"
    val realPathToWeights = OnHeapDataset::class.java.classLoader.getResource(pathToWeights).path.toString()
    val file = File(realPathToWeights)
    return HdfFile(file)
}




