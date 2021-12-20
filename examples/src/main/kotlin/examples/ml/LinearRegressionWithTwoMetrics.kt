/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.ml

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.LossFunction
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.MAE
import org.jetbrains.kotlinx.dl.api.core.metric.MSE
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000
private const val EPOCHS = 100
private const val TRAINING_BATCH_SIZE = 100

private val model = Sequential.of(
    Input(4),
    Dense(1, Activations.Linear, kernelInitializer = GlorotNormal(), biasInitializer = Zeros(), name = "dense_2")
)

/**
 * This example shows how to do regression from scratch, starting from generated dataset, using simple Dense-based [model] with 1 neuron.
 *
 * It includes:
 * - dataset creation
 * - dataset splitting
 * - model compilation
 * - model training
 * - model evaluation
 * - model weights printing
 */
fun linearRegressionWithTwoMetrics() {
    val rnd = Random(SEED)
    val data = Array(10000) { doubleArrayOf(0.0, 0.0, 0.0, 0.0, 0.0) }
    for (i in data.indices) {
        data[i][1] = 2 * (rnd.nextDouble() - 0.5)
        data[i][2] = 2 * (rnd.nextDouble() - 0.5)
        data[i][3] = 2 * (rnd.nextDouble() - 0.5)
        data[i][4] = 2 * (rnd.nextDouble() - 0.5)
        data[i][0] = data[i][1] - 2 * data[i][2] + 1.5 * data[i][3] - 0.95 * data[i][4] + rnd.nextDouble(0.1)
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

    fun extractY(): FloatArray {
        val labels = FloatArray(data.size) { 0.0f }
        for (i in labels.indices) {
            labels[i] = data[i][0].toFloat()
        }

        return labels
    }

    val dataset = OnHeapDataset.create(
        ::extractX,
        ::extractY
    )

    val (train, test) = dataset.split(0.9)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = org.jetbrains.kotlinx.dl.api.core.loss.MSE(),
            metrics = listOf(MAE(), MSE())
        )

        it.logSummary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)


        repeat(100) { id ->
            val xReal = test.getX(id)
            val yReal = test.getY(id)
            val yPred = it.predictSoftly(xReal)

            println("xReal: ${arrayOf(xReal).contentDeepToString()}, yReal: $yReal, yPred: ${yPred[0]}")
        }

        val mae = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MAE]
        val mse = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.MSE]
        println("Weights: " + it.getLayer("dense_2").weights["dense_2_dense_kernel"].contentDeepToString())
        println("Bias: " + it.getLayer("dense_2").weights["dense_2_dense_bias"].contentDeepToString())
        println("MAE: $mae")
        println("MSE: $mse")

        repeat(100) { id ->
            val xReal = test.getX(id)
            val yReal = test.getY(id)

            val yPred = it.predictSoftly(xReal)

            println("xReal: ${xReal[0]}, yReal: $yReal, yPred: ${yPred[0]}")
        }
    }
}

/** */
fun main(): Unit = linearRegressionWithTwoMetrics()


