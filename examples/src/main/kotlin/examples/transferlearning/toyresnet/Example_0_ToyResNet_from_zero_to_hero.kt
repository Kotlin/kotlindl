/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.transferlearning.toyresnet


import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import java.io.File

/**
 * We described ToyResNet in Keras and saved model configuration.
 *
 * It used simple initializers and training from zero is too long.
 *
 * It's better to load pretrained model.
 */
fun main() {
    val (train, test) = fashionMnist()

    val jsonConfigFile = getJSONConfigFileToyResNet()
    val model = Functional.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = RMSProp(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.init()
        var accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy before: $accuracy")

        it.fit(dataset = train, epochs = 3, batchSize = 100)

        accuracy = it.evaluate(dataset = test, batchSize = 1000).metrics[Metrics.ACCURACY]

        println("Accuracy after: $accuracy")
    }
}

/** Returns JSON file with model configuration, saved from Keras 2.x. */
internal fun getJSONConfigFileToyResNet(): File {
    val pathToConfig = "models/toyresnet/modelConfig.json"
    val realPathToConfig = OnHeapDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    return File(realPathToConfig)
}

/** Returns .h5 file with model weights, saved from Keras 2.x. */
internal fun getWeightsFileToyResNet(): HdfFile {
    val pathToWeights = "models/toyresnet/weights.h5"
    val realPathToWeights = OnHeapDataset::class.java.classLoader.getResource(pathToWeights).path.toString()
    return HdfFile(File(realPathToWeights))
}

