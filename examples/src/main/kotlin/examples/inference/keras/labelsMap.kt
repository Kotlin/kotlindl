/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.inference.keras

import examples.keras.cifar10.extractCifar10Images
import examples.keras.cifar10.extractCifar10Labels
import examples.production.getLabel
import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.keras.loadWeights
import org.jetbrains.kotlinx.dl.datasets.Dataset
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File

val labelsMap = mapOf(
    0 to "airplane",
    1 to "automobile",
    2 to "bird",
    3 to "cat",
    4 to "deer",
    5 to "dog",
    6 to "frog",
    7 to "horse",
    8 to "ship",
    9 to "truck"
)

val PATH_TO_MODEL_JSON = "C:\\model_1.json"
val PATH_TO_WEIGHTS = "C:\\weights.h5"
val PATH_TO_IMAGE = "C:\\zaleslaw\\home\\data\\cifar10\\images\\images\\1.png"
val imageArray = ImageConverter.toNormalizedFloatArray(File(PATH_TO_IMAGE))

const val IMAGES_ARCHIVE = "C:\\zaleslaw\\home\\data\\cifar10\\images\\images"
const val LABELS_ARCHIVE = "C:\\zaleslaw\\home\\data\\cifar10\\trainLabels.csv"

fun main() {
    val dataset = Dataset.create(
        IMAGES_ARCHIVE,
        LABELS_ARCHIVE,
        10,
        ::extractCifar10Images,
        ::extractCifar10Labels
    )

    val JSONConfig = File(PATH_TO_MODEL_JSON)
    val weights = File(PATH_TO_WEIGHTS)
    val model = Sequential.loadModelConfiguration(JSONConfig)
    model.use {
        it.compile(Adam(), Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS, Metrics.ACCURACY)
        it.summary()

        val hdfFile = HdfFile(weights)
        it.loadWeights(hdfFile)
        val prediction = it.predict(imageArray)
        println("Predicted label is: ${prediction}. This corresponds to class ${labelsMap[prediction]}.")

        for (i in 0..1000) {
            println("Predicted: ${it.predict(dataset.getX(i))} and label: ${getLabel(dataset, i)}")
        }

        val accuracy = it.evaluate(dataset = dataset, batchSize = 100).metrics[Metrics.ACCURACY]
        println("Accuracy: $accuracy")
    }
}
