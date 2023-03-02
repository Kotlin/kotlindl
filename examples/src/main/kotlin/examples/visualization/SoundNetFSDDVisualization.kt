/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.visualization

import examples.cnn.fsdd.soundBlock
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.dataset.audio.wav.WavFile
import org.jetbrains.kotlinx.dl.dataset.embedded.FSDD_SOUND_DATA_SIZE
import org.jetbrains.kotlinx.dl.dataset.embedded.NUMBER_OF_CLASSES
import org.jetbrains.kotlinx.dl.dataset.embedded.freeSpokenDigitDatasetPath
import org.jetbrains.kotlinx.dl.dataset.embedded.freeSpokenDigits
import org.jetbrains.kotlinx.dl.impl.summary.logSummary
import org.jetbrains.kotlinx.dl.visualization.letsplot.columnPlot
import org.jetbrains.kotlinx.dl.visualization.letsplot.soundPlot
import java.io.File

private const val NUM_CHANNELS = 1L
private const val SEED = 12L
private const val EPOCHS = 10
private const val TRAINING_BATCH_SIZE = 512
private const val TEST_BATCH_SIZE = 512


/**
 * This examples demonstrates model activations and Conv1D filters visualization.
 * Additionally, we present the visualization of sound files as the plots of the sound data.
 *
 * Model is trained on Free Spoken Digits Dataset.
 */
fun main() {

    visualizeFSDDWavFiles()

    val (train, test) = freeSpokenDigits()

    smallSoundNet.use {

        it.compile(
            optimizer = Adam(),
            loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
            metric = Metrics.ACCURACY
        )

        it.logSummary()

        it.fit(
            dataset = train,
            validationRate = 0.1,
            epochs = EPOCHS,
            trainBatchSize = TRAINING_BATCH_SIZE,
            validationBatchSize = TEST_BATCH_SIZE
        )

        val accuracy = it.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]
        println("Accuracy $accuracy")
    }
}

/** Create visualization plots of input sound files for selected data from Free Spoken Digits Dataset */
private fun visualizeFSDDWavFiles(
    personName: String = "george",
    samples: Int = 5
) {
    val fssd = freeSpokenDigitDatasetPath()
    val wavFiles = File(fssd).listFiles() ?: arrayOf()
    val grouped = wavFiles
        .filter { it.name.split("_")[1] == personName }
        .groupBy { it.name.split("_")[0] }
        .mapValues { it.value.sorted() }

    val plots = listOf(
        grouped["0"]!!.take(samples),
        grouped["3"]!!.take(samples),
        grouped["6"]!!.take(samples),
        grouped["9"]!!.take(samples)
    ).transpose().flatten()
        .map(::WavFile)
        .map { soundPlot(it, beginDrop = 0.01) }

    columnPlot(plots, 4, 320).show()
}

/** Transpose list of lists when every element of list is a list of equal length */
private fun <T> List<List<T>>.transpose(): List<List<T>> {
    val size = getOrNull(0)?.size ?: 0
    require(all { it.size == size }) { "Can transpose only list of lists of equal sizes" }
    return List(size) { x ->
        List(this.size) { y ->
            this[y][x]
        }
    }
}

private val smallSoundNet = Sequential.of(
    Input(
        FSDD_SOUND_DATA_SIZE,
        NUM_CHANNELS
    ),
    *soundBlock(
        filters = 16,
        kernelSize = 64,
        poolStride = 32
    ),
    *soundBlock(
        filters = 16,
        kernelSize = 128,
        poolStride = 64
    ),
    Flatten(),
    Dense(
        outputSize = 1024,
        activation = Activations.Relu,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED)
    ),
    Dense(
        outputSize = NUMBER_OF_CLASSES,
        activation = Activations.Linear,
        kernelInitializer = HeNormal(SEED),
        biasInitializer = HeNormal(SEED)
    )
)
