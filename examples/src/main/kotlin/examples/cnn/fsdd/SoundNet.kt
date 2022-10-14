/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.fsdd

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv1D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool1D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.embedded.FSDD_SOUND_DATA_SIZE
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.embedded.freeSpokenDigits

private const val EPOCHS = 20 // 20, at least, is recommended
private const val TRAINING_BATCH_SIZE = 64
private const val TEST_BATCH_SIZE = 500
private const val NUM_CHANNELS = 1L
private const val SEED = 12L

/**
 * Create a single building block for the SoundNet to simplify its structure.
 * Single block consists of two identical [Conv1D] layers followed by [MaxPool1D].
 *
 * @param filters number of filters in conv layers
 * @param kernelSize in conv layers
 * @param poolStride stride for poolSize and stride in maxpooling layer
 * @return array of layers to be registered in [Sequential] as vararg
 */
internal fun soundBlock(filters: Int, kernelSize: Int, poolStride: Int): Array<Layer> =
    arrayOf(
        Conv1D(
            filters = filters,
            kernelLength = kernelSize,
            strides = intArrayOf(1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            padding = ConvPadding.SAME
        ),
        Conv1D(
            filters = filters,
            kernelLength = kernelSize,
            strides = intArrayOf(1, 1, 1),
            activation = Activations.Relu,
            kernelInitializer = HeNormal(SEED),
            biasInitializer = HeNormal(SEED),
            padding = ConvPadding.SAME
        ),
        MaxPool1D(
            poolSize = intArrayOf(1, poolStride, 1),
            strides = intArrayOf(1, poolStride, 1),
            padding = ConvPadding.SAME
        )
    )

/**
 * This is an CNN that uses only 1D parts for convolutions and max pooling of the input sound data.
 * This network should achieve ~55% of accuracy on test data from FSDD after 10 epochs and ~85% after
 * 100 epochs.
 */
private val soundNet = Sequential.of(
    Input(
        FSDD_SOUND_DATA_SIZE,
        NUM_CHANNELS
    ),
    *soundBlock(
        filters = 4,
        kernelSize = 8,
        poolStride = 2
    ),
    *soundBlock(
        filters = 4,
        kernelSize = 16,
        poolStride = 4
    ),
    *soundBlock(
        filters = 8,
        kernelSize = 16,
        poolStride = 4
    ),
    *soundBlock(
        filters = 8,
        kernelSize = 16,
        poolStride = 4
    ),
    Flatten(),
    Dense(
        outputSize = 1024,
        activation = Activations.Relu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED)
    ),
    Dense(
        outputSize = NUMBER_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED)
    )
)

/**
 * This example shows how to do audio classification from scratch using only Conv1D layers (without Conv2D)
 * and dense layers on the example of some toy network.
 * We demonstrate the workflow on the Free Spoken Digits Dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - model compilation
 * - model training
 * - model evaluation
 */
fun soundNet() {
    val (train, test) = freeSpokenDigits()
    train.shuffle()

    soundNet.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.init()

        var accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy before: $accuracy")

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy after: $accuracy")
    }
}

fun main(): Unit = soundNet()
