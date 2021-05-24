/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.visualisation

import examples.inference.lenet5
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.fashionMnist

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

/**
 * This examples demonstrates model activations and Conv2D filters visualisation.
 *
 * Model is trained on FashionMnist dataset.
 */
fun main() {
    val (train, test) = fashionMnist()

    val (newTrain, validation) = train.split(0.95)

    val sampleIndex = 42
    val x = test.getX(sampleIndex)
    val y = test.getY(sampleIndex).toInt()

    lenet5().use { model ->

        model.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        model.fit(
            trainingDataset = newTrain,
            validationDataset = validation,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val fashionPlots = List(3) { tileImagePlot(it, test, model::predict) }
        columnPlot(fashionPlots, 3, 256).show()

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")

        val fstConv2D = model.layers[1] as Conv2D
        val sndConv2D = model.layers[3] as Conv2D

        filtersPlot(fstConv2D, columns = 16).show()
        filtersPlot(sndConv2D, columns = 16).show()

        val layersActivations = modelActivationOnLayersPlot(model, x)
        val prediction = model.predict(x)
        println("Prediction: ${fashionMnistLabelEncoding[prediction]}")
        println("Ground Truth: ${fashionMnistLabelEncoding[y]}")

        layersActivations[0].show()
        layersActivations[1].show()
    }
}
