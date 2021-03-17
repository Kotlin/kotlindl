/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.production.optimizers

import examples.inference.production.lenet5
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.datasets.mnist
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/lenet5KerasWithOptimizers"
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000

/**
 * This examples demonstrates model, model weights and optimizer weights export and import back.
 *
 * Models is exported in json format, weights are exported in custom (txt) format.
 *
 * It saves all the data to the project root directory.
 */
fun main() {
    val (train, test) = mnist()

    val (newTrain, validation) = train.split(0.95)

    val optimizer = Adam()

    lenet5.use {
        it.compile(optimizer = optimizer, loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.summary()

        print(it.kGraph())

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        it.save(
            modelDirectory = File(PATH_TO_MODEL),
            saveOptimizerState = true,
            savingFormat = SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
            writingMode = WritingMode.OVERRIDE
        )

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")
    }

    val model = Sequential.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    model.use {
        // Freeze conv2d layers, keep dense layers trainable
        for (layer in it.layers) {
            if (layer::class == Conv2D::class)
                layer.isTrainable = false
        }

        it.compile(
            optimizer = optimizer,
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()
        print(it.kGraph())
        it.loadWeights(File(PATH_TO_MODEL), loadOptimizerState = true)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 1,
            trainBatchSize = 1000,
            validationBatchSize = 100
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training with restored optimizer: $accuracyAfterTraining")
    }

    val model2 = Sequential.loadModelConfiguration(File("$PATH_TO_MODEL/modelConfig.json"))

    model2.use {
        it.compile(
            optimizer = optimizer,
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.summary()
        it.loadWeights(File(PATH_TO_MODEL), loadOptimizerState = false)

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 1,
            trainBatchSize = 1000,
            validationBatchSize = 100
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training with new optimizer: $accuracyAfterTraining")
    }
}
