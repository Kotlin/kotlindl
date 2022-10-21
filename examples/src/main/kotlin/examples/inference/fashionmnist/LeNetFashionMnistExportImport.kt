/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.fashionmnist

import examples.inference.lenet5
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel
import org.jetbrains.kotlinx.dl.dataset.embedded.fashionMnist
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/fashionLenet"
private const val EPOCHS = 3
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

/**
 * This examples demonstrates model and model weights export and import back:
 * - Model is exported as graph in .pb format, weights are exported in custom (txt) format.
 * - Model is trained on FashionMnist dataset.
 * - It saves all the data to the project root directory.
 * - [TensorFlowInferenceModel] is created via graph and weights loading.
 * - [TensorFlowInferenceModel] is reshaped and evaluated on first 10'000 images.
 */
fun lenetOnFashionMnistExportImportToTxt() {
    val (train, test) = fashionMnist()

    val (newTrain, validation) = train.split(0.95)

    lenet5().use {
        it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy $accuracy")

        it.save(File(PATH_TO_MODEL), writingMode = WritingMode.OVERRIDE)
    }

    val inferenceModel = TensorFlowInferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    inferenceModel.use {
        it.reshape(28, 28, 1)

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getX(imageId))

            if (prediction == train.getY(imageId).toInt())
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")
    }
}

/** */
fun main(): Unit = lenetOnFashionMnistExportImportToTxt()
