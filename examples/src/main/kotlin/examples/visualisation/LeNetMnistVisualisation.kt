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
import org.jetbrains.kotlinx.dl.dataset.mnist

private const val EPOCHS = 1
private const val TRAINING_BATCH_SIZE = 500
private const val TEST_BATCH_SIZE = 1000

/**
 * This examples demonstrates model activations and Conv2D filters visualisation.
 *
 * Model is trained on Mnist dataset.
 */
fun main() {

    val (train, test) = mnist()

    val sampleIndex = 42
    val x = test.getX(sampleIndex)
    val y = test.getY(sampleIndex).toInt()

    lenet5().use { model ->

        model.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        model.summary()

        model.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val numbersPlots = List(3) { tileImagePlot(it, test, model::predict) }
        columnPlot(numbersPlots, 3, 256).show()

        val accuracy = model.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")

        val fstConv2D = model.layers[1] as Conv2D
        val sndConv2D = model.layers[3] as Conv2D

        filtersPlot(fstConv2D, columns = 16).show()
        filtersPlot(sndConv2D, columns = 16).show()

        val layersActivations = modelActivationOnLayersPlot(model, x)
        val prediction = model.predict(x)
        println("Prediction: $prediction")
        println("Ground Truth: $y")

        layersActivations[0].show()
        layersActivations[1].show()
    }
}
