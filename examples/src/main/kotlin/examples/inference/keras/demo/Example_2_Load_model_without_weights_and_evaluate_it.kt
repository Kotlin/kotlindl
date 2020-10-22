/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras.demo

import api.core.Sequential
import api.core.loss.Losses
import api.core.metric.Metrics
import api.core.optimizer.Adam
import datasets.Dataset
import datasets.handlers.*

/** The model is loaded, weights are initialized by model initializers, the model just evaluated */
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

    val jsonConfigFile = getJSONConfigFile()
    val model = Sequential.loadModelConfiguration(jsonConfigFile)

    model.use {
        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )
        it.init()
        it.summary()

        val accuracy = it.evaluate(dataset = test, batchSize = 100).metrics[Metrics.ACCURACY]

        println("Accuracy training $accuracy")
    }
}





