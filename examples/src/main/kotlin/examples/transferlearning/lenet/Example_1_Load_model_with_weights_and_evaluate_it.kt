/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.lenet

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
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
 * - Weights are loaded from .h5 file, configuration is loaded from .json file.
 * - Model is evaluated after loading to obtain accuracy value.
 * - No additional training.
 * - No new layers are added.
 *
 * NOTE: Model and weights are resources in `examples` module.
 */
fun loadModelWithWeightsAndEvaluate() {
    val (_, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val hdfFile = getWeightsFile()
        it.loadWeights(hdfFile)

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy training $accuracy")
    }
}

/** */
fun main(): Unit = loadModelWithWeightsAndEvaluate()

/** Returns JSON file with model configuration, saved from Keras 2.x. */
fun getJSONConfigFile(): File {
    val pathToConfig = "models/mnist/lenet/modelConfig.json"
    val realPathToConfig = OnHeapDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    return File(realPathToConfig)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
fun getWeightsFile(): HdfFile {
    val pathToWeights = "models/mnist/lenet/mnist_weights_only.h5"
    val realPathToWeights = OnHeapDataset::class.java.classLoader.getResource(pathToWeights).path.toString()
    val file = File(realPathToWeights)
    return HdfFile(file)
}




