/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.ml

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.weights
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import kotlin.math.sin
import kotlin.random.Random

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100
private const val EPOCHS = 100
private const val TRAINING_BATCH_SIZE = 100

private val model = Sequential.of(
    Input(1, name = "input_1"),
    Dense(
        20,
        Activations.Relu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeUniform(SEED),
        name = "dense_1"
    ),
    Dense(
        20,
        Activations.Relu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeUniform(SEED),
        name = "dense_2"
    ),
    Dense(1, Activations.Linear, name = "dense_3")
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
fun sineRegression() {
    val (train, test) = prepareDataset().split(0.8)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.MAE,
            metric = Metrics.MAE
        )

        it.logSummary()

        it.fit(
            dataset = train,
            epochs = EPOCHS,
            batchSize = TRAINING_BATCH_SIZE
        )

        val evaluationResult = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE)

        println("evaluationResult: $evaluationResult")

        println("Weights: " + it.getLayer("dense_3").weights["dense_3_dense_kernel"].contentDeepToString())
        println("Bias: " + it.getLayer("dense_3").weights["dense_3_dense_bias"].contentDeepToString())

        repeat(100) { id ->
            val xReal = test.getX(id)
            val yReal = test.getY(id)

            val yPred = it.predictSoftly(xReal)

            println("xReal: ${xReal[0]}, yReal: $yReal, yPred: ${yPred[0]}")
        }
    }
}

fun prepareDataset(): Dataset {
    val sampleCount = 100000

    val x = Array(sampleCount) { FloatArray(1) }
    val y = FloatArray(sampleCount)

    repeat(sampleCount) {
        val xSample = Random.nextDouble(0.0, Math.PI * 2).toFloat()
        val ySample = sin(xSample)

        x[it][0] = xSample
        y[it] = ySample
    }

    return OnHeapDataset.create(x, y)
}

/** */
fun main(): Unit = sineRegression()
