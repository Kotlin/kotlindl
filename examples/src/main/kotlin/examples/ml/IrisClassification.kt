/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.ml

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

private const val SEED = 12L
private const val TEST_BATCH_SIZE = 5
private const val EPOCHS = 100
private const val TRAINING_BATCH_SIZE = 5

private val model = Sequential.of(
    Input(4),
    Dense(300, Activations.Relu, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros()),
    Dense(3, Activations.Linear, kernelInitializer = HeNormal(SEED), biasInitializer = Zeros())
)

/**
 * This example shows how to do classification from scratch, starting from static Iris dataset, using simple Dense-based [model].
 *
 * It includes:
 * - dataset creation
 * - dataset splitting
 * - model compilation
 * - model training
 * - model evaluation
 */
fun irisClassification() {
    data.shuffle()

    val dataset = OnHeapDataset.create(
        ::extractX,
        ::extractY
    )

    val (train, test) = dataset.split(0.9)

    model.use {
        it.compile(optimizer = SGD(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.logSummary()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = irisClassification()

private fun extractX(): Array<FloatArray> {
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

private fun extractY(): FloatArray {
    val labels = FloatArray(data.size) { 0.0f }
    for (i in labels.indices) {
        val classLabel = data[i][0]
        labels[i] = classLabel.toFloat()
    }

    return labels
}

private val data = arrayOf(
    doubleArrayOf(0.0, 5.1, 3.5, 1.4, 0.2),
    doubleArrayOf(0.0, 4.9, 3.0, 1.4, 0.2),
    doubleArrayOf(0.0, 4.7, 3.2, 1.3, 0.2),
    doubleArrayOf(0.0, 4.6, 3.1, 1.5, 0.2),
    doubleArrayOf(0.0, 5.0, 3.6, 1.4, 0.2),
    doubleArrayOf(0.0, 5.4, 3.9, 1.7, 0.4),
    doubleArrayOf(0.0, 4.6, 3.4, 1.4, 0.3),
    doubleArrayOf(0.0, 5.0, 3.4, 1.5, 0.2),
    doubleArrayOf(0.0, 4.4, 2.9, 1.4, 0.2),
    doubleArrayOf(0.0, 4.9, 3.1, 1.5, 0.1),
    doubleArrayOf(0.0, 5.4, 3.7, 1.5, 0.2),
    doubleArrayOf(0.0, 4.8, 3.4, 1.6, 0.2),
    doubleArrayOf(0.0, 4.8, 3.0, 1.4, 0.1),
    doubleArrayOf(0.0, 4.3, 3.0, 1.1, 0.1),
    doubleArrayOf(0.0, 5.8, 4.0, 1.2, 0.2),
    doubleArrayOf(0.0, 5.7, 4.4, 1.5, 0.4),
    doubleArrayOf(0.0, 5.4, 3.9, 1.3, 0.4),
    doubleArrayOf(0.0, 5.1, 3.5, 1.4, 0.3),
    doubleArrayOf(0.0, 5.7, 3.8, 1.7, 0.3),
    doubleArrayOf(0.0, 5.1, 3.8, 1.5, 0.3),
    doubleArrayOf(0.0, 5.4, 3.4, 1.7, 0.2),
    doubleArrayOf(0.0, 5.1, 3.7, 1.5, 0.4),
    doubleArrayOf(0.0, 4.6, 3.6, 1.0, 0.2),
    doubleArrayOf(0.0, 5.1, 3.3, 1.7, 0.5),
    doubleArrayOf(0.0, 4.8, 3.4, 1.9, 0.2),
    doubleArrayOf(0.0, 5.0, 3.0, 1.6, 0.2),
    doubleArrayOf(0.0, 5.0, 3.4, 1.6, 0.4),
    doubleArrayOf(0.0, 5.2, 3.5, 1.5, 0.2),
    doubleArrayOf(0.0, 5.2, 3.4, 1.4, 0.2),
    doubleArrayOf(0.0, 4.7, 3.2, 1.6, 0.2),
    doubleArrayOf(0.0, 4.8, 3.1, 1.6, 0.2),
    doubleArrayOf(0.0, 5.4, 3.4, 1.5, 0.4),
    doubleArrayOf(0.0, 5.2, 4.1, 1.5, 0.1),
    doubleArrayOf(0.0, 5.5, 4.2, 1.4, 0.2),
    doubleArrayOf(0.0, 4.9, 3.1, 1.5, 0.1),
    doubleArrayOf(0.0, 5.0, 3.2, 1.2, 0.2),
    doubleArrayOf(0.0, 5.5, 3.5, 1.3, 0.2),
    doubleArrayOf(0.0, 4.9, 3.1, 1.5, 0.1),
    doubleArrayOf(0.0, 4.4, 3.0, 1.3, 0.2),
    doubleArrayOf(0.0, 5.1, 3.4, 1.5, 0.2),
    doubleArrayOf(0.0, 5.0, 3.5, 1.3, 0.3),
    doubleArrayOf(0.0, 4.5, 2.3, 1.3, 0.3),
    doubleArrayOf(0.0, 4.4, 3.2, 1.3, 0.2),
    doubleArrayOf(0.0, 5.0, 3.5, 1.6, 0.6),
    doubleArrayOf(0.0, 5.1, 3.8, 1.9, 0.4),
    doubleArrayOf(0.0, 4.8, 3.0, 1.4, 0.3),
    doubleArrayOf(0.0, 5.1, 3.8, 1.6, 0.2),
    doubleArrayOf(0.0, 4.6, 3.2, 1.4, 0.2),
    doubleArrayOf(0.0, 5.3, 3.7, 1.5, 0.2),
    doubleArrayOf(0.0, 5.0, 3.3, 1.4, 0.2),
    doubleArrayOf(1.0, 7.0, 3.2, 4.7, 1.4),
    doubleArrayOf(1.0, 6.4, 3.2, 4.5, 1.5),
    doubleArrayOf(1.0, 6.9, 3.1, 4.9, 1.5),
    doubleArrayOf(1.0, 5.5, 2.3, 4.0, 1.3),
    doubleArrayOf(1.0, 6.5, 2.8, 4.6, 1.5),
    doubleArrayOf(1.0, 5.7, 2.8, 4.5, 1.3),
    doubleArrayOf(1.0, 6.3, 3.3, 4.7, 1.6),
    doubleArrayOf(1.0, 4.9, 2.4, 3.3, 1.0),
    doubleArrayOf(1.0, 6.6, 2.9, 4.6, 1.3),
    doubleArrayOf(1.0, 5.2, 2.7, 3.9, 1.4),
    doubleArrayOf(1.0, 5.0, 2.0, 3.5, 1.0),
    doubleArrayOf(1.0, 5.9, 3.0, 4.2, 1.5),
    doubleArrayOf(1.0, 6.0, 2.2, 4.0, 1.0),
    doubleArrayOf(1.0, 6.1, 2.9, 4.7, 1.4),
    doubleArrayOf(1.0, 5.6, 2.9, 3.6, 1.3),
    doubleArrayOf(1.0, 6.7, 3.1, 4.4, 1.4),
    doubleArrayOf(1.0, 5.6, 3.0, 4.5, 1.5),
    doubleArrayOf(1.0, 5.8, 2.7, 4.1, 1.0),
    doubleArrayOf(1.0, 6.2, 2.2, 4.5, 1.5),
    doubleArrayOf(1.0, 5.6, 2.5, 3.9, 1.1),
    doubleArrayOf(1.0, 5.9, 3.2, 4.8, 1.8),
    doubleArrayOf(1.0, 6.1, 2.8, 4.0, 1.3),
    doubleArrayOf(1.0, 6.3, 2.5, 4.9, 1.5),
    doubleArrayOf(1.0, 6.1, 2.8, 4.7, 1.2),
    doubleArrayOf(1.0, 6.4, 2.9, 4.3, 1.3),
    doubleArrayOf(1.0, 6.6, 3.0, 4.4, 1.4),
    doubleArrayOf(1.0, 6.8, 2.8, 4.8, 1.4),
    doubleArrayOf(1.0, 6.7, 3.0, 5.0, 1.7),
    doubleArrayOf(1.0, 6.0, 2.9, 4.5, 1.5),
    doubleArrayOf(1.0, 5.7, 2.6, 3.5, 1.0),
    doubleArrayOf(1.0, 5.5, 2.4, 3.8, 1.1),
    doubleArrayOf(1.0, 5.5, 2.4, 3.7, 1.0),
    doubleArrayOf(1.0, 5.8, 2.7, 3.9, 1.2),
    doubleArrayOf(1.0, 6.0, 2.7, 5.1, 1.6),
    doubleArrayOf(1.0, 5.4, 3.0, 4.5, 1.5),
    doubleArrayOf(1.0, 6.0, 3.4, 4.5, 1.6),
    doubleArrayOf(1.0, 6.7, 3.1, 4.7, 1.5),
    doubleArrayOf(1.0, 6.3, 2.3, 4.4, 1.3),
    doubleArrayOf(1.0, 5.6, 3.0, 4.1, 1.3),
    doubleArrayOf(1.0, 5.5, 2.5, 4.0, 1.3),
    doubleArrayOf(1.0, 5.5, 2.6, 4.4, 1.2),
    doubleArrayOf(1.0, 6.1, 3.0, 4.6, 1.4),
    doubleArrayOf(1.0, 5.8, 2.6, 4.0, 1.2),
    doubleArrayOf(1.0, 5.0, 2.3, 3.3, 1.0),
    doubleArrayOf(1.0, 5.6, 2.7, 4.2, 1.3),
    doubleArrayOf(1.0, 5.7, 3.0, 4.2, 1.2),
    doubleArrayOf(1.0, 5.7, 2.9, 4.2, 1.3),
    doubleArrayOf(1.0, 6.2, 2.9, 4.3, 1.3),
    doubleArrayOf(1.0, 5.1, 2.5, 3.0, 1.1),
    doubleArrayOf(1.0, 5.7, 2.8, 4.1, 1.3),
    doubleArrayOf(2.0, 6.3, 3.3, 6.0, 2.5),
    doubleArrayOf(2.0, 5.8, 2.7, 5.1, 1.9),
    doubleArrayOf(2.0, 7.1, 3.0, 5.9, 2.1),
    doubleArrayOf(2.0, 6.3, 2.9, 5.6, 1.8),
    doubleArrayOf(2.0, 6.5, 3.0, 5.8, 2.2),
    doubleArrayOf(2.0, 7.6, 3.0, 6.6, 2.1),
    doubleArrayOf(2.0, 4.9, 2.5, 4.5, 1.7),
    doubleArrayOf(2.0, 7.3, 2.9, 6.3, 1.8),
    doubleArrayOf(2.0, 6.7, 2.5, 5.8, 1.8),
    doubleArrayOf(2.0, 7.2, 3.6, 6.1, 2.5),
    doubleArrayOf(2.0, 6.5, 3.2, 5.1, 2.0),
    doubleArrayOf(2.0, 6.4, 2.7, 5.3, 1.9),
    doubleArrayOf(2.0, 6.8, 3.0, 5.5, 2.1),
    doubleArrayOf(2.0, 5.7, 2.5, 5.0, 2.0),
    doubleArrayOf(2.0, 5.8, 2.8, 5.1, 2.4),
    doubleArrayOf(2.0, 6.4, 3.2, 5.3, 2.3),
    doubleArrayOf(2.0, 6.5, 3.0, 5.5, 1.8),
    doubleArrayOf(2.0, 7.7, 3.8, 6.7, 2.2),
    doubleArrayOf(2.0, 7.7, 2.6, 6.9, 2.3),
    doubleArrayOf(2.0, 6.0, 2.2, 5.0, 1.5),
    doubleArrayOf(2.0, 6.9, 3.2, 5.7, 2.3),
    doubleArrayOf(2.0, 5.6, 2.8, 4.9, 2.0),
    doubleArrayOf(2.0, 7.7, 2.8, 6.7, 2.0),
    doubleArrayOf(2.0, 6.3, 2.7, 4.9, 1.8),
    doubleArrayOf(2.0, 6.7, 3.3, 5.7, 2.1),
    doubleArrayOf(2.0, 7.2, 3.2, 6.0, 1.8),
    doubleArrayOf(2.0, 6.2, 2.8, 4.8, 1.8),
    doubleArrayOf(2.0, 6.1, 2.0, 4.9, 1.8),
    doubleArrayOf(2.0, 6.4, 2.8, 5.6, 2.1),
    doubleArrayOf(2.0, 7.2, 3.0, 5.8, 1.6),
    doubleArrayOf(2.0, 7.4, 2.8, 6.1, 1.9),
    doubleArrayOf(2.0, 7.9, 3.8, 6.4, 2.0),
    doubleArrayOf(2.0, 6.4, 2.8, 5.6, 2.2),
    doubleArrayOf(2.0, 6.3, 2.8, 5.1, 1.5),
    doubleArrayOf(2.0, 6.1, 2.6, 5.6, 1.4),
    doubleArrayOf(2.0, 7.7, 3.0, 6.1, 2.3),
    doubleArrayOf(2.0, 6.3, 3.4, 5.6, 2.4),
    doubleArrayOf(2.0, 6.4, 3.1, 5.5, 1.8),
    doubleArrayOf(2.0, 6.0, 3.0, 4.8, 1.8),
    doubleArrayOf(2.0, 6.9, 3.1, 5.4, 2.1),
    doubleArrayOf(2.0, 6.7, 3.1, 5.6, 2.4),
    doubleArrayOf(2.0, 6.9, 3.1, 5.1, 2.3),
    doubleArrayOf(2.0, 5.8, 2.7, 5.1, 1.9),
    doubleArrayOf(2.0, 6.8, 3.2, 5.9, 2.3),
    doubleArrayOf(2.0, 6.7, 3.3, 5.7, 2.5),
    doubleArrayOf(2.0, 6.7, 3.0, 5.2, 2.3),
    doubleArrayOf(2.0, 6.3, 2.5, 5.0, 1.9),
    doubleArrayOf(2.0, 6.5, 3.0, 5.2, 2.0),
    doubleArrayOf(2.0, 6.2, 3.4, 5.4, 2.3),
    doubleArrayOf(2.0, 5.9, 3.0, 5.1, 1.8)
)
