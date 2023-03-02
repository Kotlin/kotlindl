/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.embedded

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import java.io.File
import java.io.IOException

private const val DATASET_SIZE = 50000

/** Loads images from [archiveName] to heap memory and applies basic normalization preprocessing. */
@Throws(IOException::class)
public fun extractCifar10Images(archiveName: String): Array<FloatArray> {
    return loadImagesFromDirectory(
        DATASET_SIZE,
        archiveName
    )
}

private fun loadImagesFromDirectory(
    subDatasetSize: Int,
    archiveName: String
): Array<FloatArray> {
    val images = Array(subDatasetSize) {
        ImageConverter.toNormalizedFloatArray(File(archiveName, "${it + 1}.png"), colorMode = ColorMode.BGR)
    }

    return images
}

/** Loads labels from [pathToLabels] csv file to heap memory and converts to Floats. */
@Throws(IOException::class)
public fun extractCifar10Labels(pathToLabels: String): FloatArray {
    val labelCount = DATASET_SIZE
    println(String.format("Extracting %d labels from %s", labelCount, pathToLabels))
    val labelBuffer = ByteArray(labelCount)

    val dictionary = mapOf(
        "airplane" to 0,
        "automobile" to 1,
        "bird" to 2,
        "cat" to 3,
        "deer" to 4,
        "dog" to 5,
        "frog" to 6,
        "horse" to 7,
        "ship" to 8,
        "truck" to 9
    )

    var cnt = 0
    csvReader().open(pathToLabels) {
        readAllAsSequence()
            .forEach { row ->
                labelBuffer[cnt] = dictionary.getOrElse(row[1]) { 1 }.toByte()
                cnt++
            }
    }

    val floats = FloatArray(labelCount)

    for (i in 0 until labelCount) {
        floats[i] = OnHeapDataset.convertByteToFloat(labelBuffer[i])
    }
    return floats
}

/**
 * Loads labels from [pathToLabels] csv file to heap memory and converts to Floats, after that it sorts it to have the same order as image files.
 *
 * NOTE: It's important if you are going to use it with [org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset].
 */
@Throws(IOException::class)
public fun extractCifar10LabelsAnsSort(pathToLabels: String): FloatArray {
    val labelCount = DATASET_SIZE
    println(String.format("Extracting %d labels from %s", labelCount, pathToLabels))
    val labelSorter = mutableMapOf<String, Int>()

    val dictionary = mapOf(
        "airplane" to 0,
        "automobile" to 1,
        "bird" to 2,
        "cat" to 3,
        "deer" to 4,
        "dog" to 5,
        "frog" to 6,
        "horse" to 7,
        "ship" to 8,
        "truck" to 9
    )

    csvReader().open(pathToLabels) {
        readAllAsSequence()
            .forEach { row ->
                labelSorter[row[0]] = dictionary.getOrElse(row[1]) { 1 }
            }
    }

    val sortedMap = labelSorter.toSortedMap()

    val labelBuffer = sortedMap.values.toIntArray()

    val floats = FloatArray(labelCount)

    for (i in 0 until labelCount) {
        floats[i] =
            OnHeapDataset.convertByteToFloat(
                labelBuffer[i].toByte()
            )
    }
    return floats
}
