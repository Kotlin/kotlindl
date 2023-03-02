/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.mnist.advanced

import examples.cnn.models.buildLetNet5Classic
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.callback.Callback
import org.jetbrains.kotlinx.dl.api.core.callback.EarlyStopping
import org.jetbrains.kotlinx.dl.api.core.callback.EarlyStoppingMode
import org.jetbrains.kotlinx.dl.api.core.callback.TerminateOnNaN
import org.jetbrains.kotlinx.dl.api.core.history.BatchEvent
import org.jetbrains.kotlinx.dl.api.core.history.EpochTrainingEvent
import org.jetbrains.kotlinx.dl.api.core.history.History
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.core.optimizer.ClipGradientByValue
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.embedded.mnist
import org.jetbrains.kotlinx.dl.impl.summary.logSummary

private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 1000
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L
private const val TEST_BATCH_SIZE = 1000

private val lenet5Classic = buildLetNet5Classic(
    imageWidth = IMAGE_SIZE,
    imageHeight = IMAGE_SIZE,
    numChannels = NUM_CHANNELS,
    numClasses = NUMBER_OF_CLASSES,
    layersActivation = Activations.Tanh,
    classifierActivation = Activations.Linear,
    randomSeed = SEED,
)

/**
 * This example shows how to do image classification from scratch using [lenet5Classic], without leveraging pre-trained weights or a pre-made model.
 * We demonstrate the workflow on the Mnist classification dataset.
 *
 * It includes:
 * - dataset loading from S3
 * - callback definition
 * - model compilation with [EarlyStopping] and [TerminateOnNaN] callbacks
 * - model summary
 * - model training
 * - model evaluation
 */
fun lenetWithMultipleCallbacks() {
    val (train, test) = mnist()

    lenet5Classic.use {
        val earlyStopping = EarlyStopping(
            monitor = EpochTrainingEvent::valLossValue,
            minDelta = 0.0,
            patience = 2,
            verbose = true,
            mode = EarlyStoppingMode.AUTO,
            baseline = 0.1,
            restoreBestWeights = false
        )
        val terminateOnNaN = TerminateOnNaN()
        it.compile(
            optimizer = Adam(clipGradient = ClipGradientByValue(0.1f)),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(
            dataset = train,
            epochs = EPOCHS,
            batchSize = TRAINING_BATCH_SIZE,
            callbacks = listOf(earlyStopping, terminateOnNaN)
        )

        val accuracy = it.evaluate(
            dataset = test,
            batchSize = TEST_BATCH_SIZE,
            callback = EvaluateCallback()
        ).metrics[Metrics.ACCURACY]


        println("Accuracy: $accuracy")

        val predictions = it.predict(
            dataset = test,
            batchSize = TEST_BATCH_SIZE,
            callback = PredictCallback()
        )

        println(predictions.size)
    }
}

/** Simple custom Callback object. */
class EvaluateCallback : Callback() {
    override fun onTestBatchBegin(batch: Int, batchSize: Int, logs: History) {
        println("Test batch $batch begins.")
    }

    override fun onTestBatchEnd(batch: Int, batchSize: Int, event: BatchEvent?, logs: History) {
        println("Test batch $batch ends with loss ${event!!.lossValue}..")
    }

    override fun onTestBegin() {
        println("Test begins")
    }

    override fun onTestEnd(logs: History) {
        println("Train ends with last loss ${logs.lastBatchEvent().lossValue}")
    }
}

/** Simple custom Callback object. */
class PredictCallback : Callback() {
    override fun onPredictBatchBegin(batch: Int, batchSize: Int) {
        println("Prediction batch $batch begins.")
    }

    override fun onPredictBatchEnd(batch: Int, batchSize: Int) {
        println("Prediction batch $batch ends.")
    }

    override fun onPredictBegin() {
        println("Predict begins")
    }

    override fun onPredictEnd() {
        println("Predict ends")
    }
}


/** */
fun main(): Unit = lenetWithMultipleCallbacks()
