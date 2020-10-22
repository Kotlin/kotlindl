/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.production

import api.core.WritingMode
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.Adam
import api.inference.InferenceModel
import datasets.Dataset
import datasets.handlers.*
import java.io.File

private const val PATH_TO_MODEL = "savedmodels/fashionLenet"
private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

private val fashionMnistLabelEncoding = mapOf(
    0 to "T-shirt/top",
    1 to "Trouser",
    2 to "Pullover",
    3 to "Dress",
    4 to "Coat",
    5 to "Sandal",
    6 to "Shirt",
    7 to "Sneaker",
    8 to "Bag",
    9 to "Ankle boot"
)

fun main() {
    val (train, test) = Dataset.createTrainAndTestDatasets(
        FASHION_TRAIN_IMAGES_ARCHIVE,
        FASHION_TRAIN_LABELS_ARCHIVE,
        FASHION_TEST_IMAGES_ARCHIVE,
        FASHION_TEST_LABELS_ARCHIVE,
        NUMBER_OF_CLASSES,
        ::extractFashionImages,
        ::extractFashionLabels
    )

    val (newTrain, validation) = train.split(0.95)

    lenet5.use {
        it.compile(optimizer = Adam(), loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, metric = Metrics.ACCURACY)

        it.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE,
            verbose = true
        )

        val weights = it.layers[0].getWeights() // first conv2d layer

        drawFilters(weights[0])

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

        println("Accuracy $accuracy")

        it.save(File(PATH_TO_MODEL), writingMode = WritingMode.OVERRIDE)
    }

    val inferenceModel = InferenceModel.load(File(PATH_TO_MODEL), loadOptimizerState = true)

    inferenceModel.use {
        it.reshape(::mnistReshape)

        var accuracy = 0.0
        val amountOfTestSet = 10000
        for (imageId in 0..amountOfTestSet) {
            val prediction = it.predict(train.getX(imageId))

            if (prediction == getLabel(train, imageId))
                accuracy += (1.0 / amountOfTestSet)
        }
        println("Accuracy: $accuracy")
    }
}