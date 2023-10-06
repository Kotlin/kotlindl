/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist.advanced

import org.jetbrains.kotlinx.dl.api.core.GpuConfiguration
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000

/**
 * This example shows how to do image classification from scratch using [lenet5Classic], without leveraging pre-trained weights or a pre-made model.
 * We demonstrate the workflow on the Mnist classification dataset.
 *
 * It could be run only with enabled tensorflow GPU dependencies
 *
 * It includes:
 * - dataset loading from S3
 * - model compilation
 * - model summary
 * - model training
 * - model evaluation
 */
fun lenetClassicWithGPUMemoryConfig() {

    val layersActivation = Activations.Tanh
    val classifierActivation = Activations.Linear

    val model = Sequential.of(
        Input(
            IMAGE_SIZE,
            IMAGE_SIZE,
            NUM_CHANNELS,
        ),
        Conv2D(
            filters = 6,
            kernelSize = 5,
            strides = 1,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
        ),
        AvgPool2D(
            poolSize = 2,
            strides = 2,
            padding = ConvPadding.VALID,
        ),
        Conv2D(
            filters = 16,
            kernelSize = 5,
            strides = 1,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Zeros(),
            padding = ConvPadding.SAME,
        ),
        AvgPool2D(
            poolSize = 2,
            strides = 2,
            padding = ConvPadding.VALID,
        ),
        Flatten(),
        Dense(
            outputSize = 120,
            activation = layersActivation,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Constant(0.1f),
        ),
        Dense(
            outputSize = 84,
            activation = Activations.Tanh,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Constant(0.1f),
        ),
        Dense(
            outputSize = NUMBER_OF_CLASSES,
            activation = classifierActivation,
            kernelInitializer = GlorotNormal(SEED),
            biasInitializer = Constant(0.1f),
        ),
        gpuConfiguration = GpuConfiguration(allowGrowth = true)
    )

    val (train, test) = mnist()

    model.use {
        it.compile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = lenetClassicWithGPUMemoryConfig()

