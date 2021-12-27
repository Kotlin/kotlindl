/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist

import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.mnist

private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 200
private const val TEST_BATCH_SIZE = 1000
private const val NUM_LABELS = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val heNormal = HeNormal(SEED)

/**
 * The implementation in Kotlin of VGG'11 model with minor changes of number of neurons to be successfully applied to the MNIST dataset.
 */
private val vgg11 = Sequential.of(
    Input(
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS
    ),
    Conv2D(
        filters = 32,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 64,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 256,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 256,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    Conv2D(
        filters = 128,
        kernelSize = intArrayOf(3, 3),
        strides = intArrayOf(1, 1, 1, 1),
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal,
        padding = ConvPadding.SAME
    ),
    MaxPool2D(
        poolSize = intArrayOf(1, 2, 2, 1),
        strides = intArrayOf(1, 2, 2, 1),
        padding = ConvPadding.SAME
    ),
    Flatten(),
    Dense(
        outputSize = 2048,
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal
    ),
    Dense(
        outputSize = 1000,
        activation = Activations.Relu,
        kernelInitializer = heNormal,
        biasInitializer = heNormal
    ),
    Dense(
        outputSize = NUM_LABELS,
        activation = Activations.Linear,
        kernelInitializer = heNormal,
        biasInitializer = heNormal
    )
)


/**
 * This example shows how to do image classification from scratch using [vgg11], without leveraging pre-trained weights or a pre-made model.
 * We demonstrate the workflow on the Mnist classification dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - model compilation
 * - model summary
 * - model training
 * - model evaluation
 */
fun vgg() {
    val (train, test) = mnist()

    vgg11.use {
        it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.logSummary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = vgg()
