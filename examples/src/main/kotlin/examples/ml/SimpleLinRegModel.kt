/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.ml

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.datasets.Dataset
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 100

private val model = Sequential.of(
    Input(4),
    Dense(1, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

fun main() {
    val rnd = Random(SEED)
    val data = Array(10000) { doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0) }
    for (i in data.indices) {
        data[i][1] = 2 * (rnd.nextDouble() - 0.5)
        data[i][2] = 2 * (rnd.nextDouble() - 0.5)
        data[i][3] = 2 * (rnd.nextDouble() - 0.5)
        data[i][4] = 2 * (rnd.nextDouble() - 0.5)
        data[i][0] = data[i][1] - 2 * data[i][2] + 1.5 * data[i][3] - 0.95 * data[i][4] + 0.2 + rnd.nextDouble(0.1)
        // 1 * x1 - 2 * x2 + 1.5 * x3  - 0.95 * x4 +- 0.1
    }

    data.shuffle()

    fun extractX(): Array<FloatArray> {
        val init: (index: Int) -> FloatArray = { index ->
            floatArrayOf(
                data[index][1].toFloat(),
                data[index][2].toFloat(),
                data[index][3].toFloat(),
                data[index][4].toFloat()
            )
        }
        return Array(data.size, init = init)
    }

    fun extractY(): Array<FloatArray> {
        val labels = Array(data.size) { FloatArray(1) { 0.0f } }
        for (i in labels.indices) {
            labels[i][0] = data[i][0].toFloat()
        }

        return labels
    }

    val dataset = Dataset.create(
        ::extractX,
        ::extractY
    )

    val (train, test) = dataset.split(0.9)

    model.use {
        it.compile(
            optimizer = SGD(learningRate = 0.001f),
            loss = Losses.MLSE,
            metric = Metrics.MLSE
        )

        it.summary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE, verbose = true)

        val mse = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MSE]
        println("Weights: " + it.getLayer("dense_1").getWeights()[0].contentDeepToString())
        println("Bias" + it.getLayer("dense_1").getWeights()[1].contentDeepToString())
        println("MSE: $mse")
    }
}


