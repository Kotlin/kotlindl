/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.mnist

import examples.inference.lenet5
import org.jetbrains.kotlinx.dl.api.core.SavingFormat
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.RMSProp
import org.jetbrains.kotlinx.dl.api.core.optimizer.SGD
import org.jetbrains.kotlinx.dl.api.core.summary.logSummary
import org.jetbrains.kotlinx.dl.dataset.mnist
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/lenet5_keras"
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 1000
private const val TEST_BATCH_SIZE = 1000

/**
 * This examples demonstrates model and model weights export and import back:
 * - Model is exported in Keras-style JSON format; weights are exported in custom (TXT) format.
 * - Model is trained on Mnist dataset.
 * - It saves all the data to the project root directory.
 * - [Sequential] model is created via JSON configuration and weights loading.
 * - After loading model is trained again with another optimizer with frozen Conv2D layers. Only weights in Dense layers can be updated.
 */
fun lenetOnMnistExportImportToJson() {
    val (train, test) = mnist()

    val (newTrain, validation) = train.split(0.95)

    lenet5().use {
        it.compile(
            optimizer = SGD(learningRate = 0.05f),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        it.save(
            File(PATH_TO_MODEL),
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES,
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
            optimizer = RMSProp(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.logSummary()

        it.loadWeights(File(PATH_TO_MODEL))

        val accuracyBefore = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy before training $accuracyBefore")

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = 2,
            trainBatchSize = 1000,
            validationBatchSize = 100
        )

        val accuracyAfterTraining = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy after training $accuracyAfterTraining")
    }
}

/** */
fun main(): Unit = lenetOnMnistExportImportToJson()
