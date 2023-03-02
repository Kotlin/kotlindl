/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.fashionmnist

import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.model.resnet50Light
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

private const val EPOCHS = 5
private const val TRAINING_BATCH_SIZE = 64
private const val TEST_BATCH_SIZE = 64
private const val NUM_CLASSES = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L

/**
 * This example shows how to do image classification from scratch using pre-made [resnet50Light] model, without leveraging pre-trained weights.
 * We demonstrate the workflow on the FashionMnist classification dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - usage of pre-defined model from [org.jetbrains.kotlinx.dl.api.core.model] package
 * - model compilation
 * - model training
 * - model evaluation
 */
fun resnet50OnFashionMnistDataset() {
    val (train, test) = fashionMnist()

    resnet50Light(imageSize = IMAGE_SIZE, numberOfClasses = NUM_CLASSES, numberOfInputChannels = NUM_CHANNELS).use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        val start = System.currentTimeMillis()
        it.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)
        println("Training time: ${(System.currentTimeMillis() - start) / 1000f}")

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = resnet50OnFashionMnistDataset()
