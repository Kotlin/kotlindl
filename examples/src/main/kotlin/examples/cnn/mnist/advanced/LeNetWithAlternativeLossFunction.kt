/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist.advanced

import examples.cnn.models.buildLetNet5Classic
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

private const val EPOCHS = 2
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 100

private val lenet5Classic = buildLetNet5Classic(
    imageWidth = IMAGE_SIZE,
    imageHeight = IMAGE_SIZE,
    numChannels = NUM_CHANNELS,
    numClasses = NUMBER_OF_CLASSES,
    layersActivation = Activations.Relu,
    classifierActivation = Activations.Softmax,
    randomSeed = SEED,
)

/**
 * This example shows how to do image classification from scratch using [lenet5Classic], without leveraging pre-trained weights or a pre-made model.
 * We demonstrate the workflow on the Mnist classification dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - dataset splitting on train, test and validation subsets
 * - model compilation with alternative [Losses]
 * - model summary
 * - model training with validation
 * - model evaluation
 */
fun lenetWithAlternativeLossFunction() {
    val (train, test) = mnist()

    val (newTrain, validation) = train.split(0.95)

    lenet5Classic.use { model ->
        model.compile(
            optimizer = Adam(),
            loss = Losses.HUBER,
            metric = Metrics.ACCURACY
        )

        model.logSummary()

        val history = model.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")

        val accuracyByEpoch = history.epochHistory.map { it.metricValues[0] }.toDoubleArray()
        println(accuracyByEpoch.contentToString())
    }
}

/** */
fun main(): Unit = lenetWithAlternativeLossFunction()
