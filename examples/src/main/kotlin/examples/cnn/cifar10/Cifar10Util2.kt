/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.cnn.cifar10

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.io.IOException

private const val DATASET_SIZE = 50000

@Throws(IOException::class)
fun extractCifar10LabelsAnsSort(pathToLabels: String, numClasses: Int): FloatArray {
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
