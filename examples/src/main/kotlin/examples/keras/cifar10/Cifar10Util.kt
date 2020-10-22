/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.keras.cifar10

import com.github.doyaaaaaken.kotlincsv.dsl.csvReader
import datasets.Dataset
import datasets.image.ImageConverter
import java.awt.image.DataBufferByte
import java.io.IOException
import java.util.zip.ZipEntry
import java.util.zip.ZipFile

const val IMAGES_ARCHIVE = "C:\\zaleslaw\\home\\data\\cifar10\\images.zip"
const val LABELS_ARCHIVE = "C:\\zaleslaw\\home\\data\\cifar10\\trainLabels.csv"
private const val DATASET_SIZE = 50000

@Throws(IOException::class)
fun extractCifar10Images(archiveName: String): Array<FloatArray> {
    val numOfPixels: Int = 32 * 32 * 3

    return loadImagesFromZipArchive(
        numOfPixels,
        DATASET_SIZE,
        archiveName
    )
}

private fun loadImagesFromZipArchive(
    numOfPixels: Int,
    subDatasetSize: Int,
    archiveName: String
): Array<FloatArray> {
    val images = Array(subDatasetSize) { FloatArray(numOfPixels) }
    val zipFile = ZipFile(archiveName)
    val entries = zipFile.entries()

    var cnt = 0

    while (entries.hasMoreElements()) {
        val entry = entries.nextElement() as ZipEntry
        val image = ImageConverter.getImage(zipFile.getInputStream(entry))

        val pixels = (image.raster.dataBuffer as DataBufferByte).data

        images[cnt] =
            Dataset.toNormalizedVector(
                pixels
            )
        cnt++
    }

    zipFile.close()
    return images
}

@Throws(IOException::class)
fun extractCifar10Labels(pathToLabels: String, numClasses: Int): Array<FloatArray> {
    val labelCount = DATASET_SIZE
    println(String.format("Extracting %d labels from %s", labelCount, pathToLabels))
    val labelBuffer = ByteArray(labelCount)

    val dictionary = mapOf(
        "airplane" to 0, "automobile" to 1, "bird" to 2, "cat" to 3, "deer" to 4, "dog" to 5, "frog" to 6,
        "horse" to 7,
        "ship" to 8,
        "truck" to 9
    )

    var cnt = 0
    csvReader().open(pathToLabels) {
        readAllAsSequence().forEach { row ->
            labelBuffer[cnt] = dictionary.getOrElse(row[1]) { 1 }.toByte()
            cnt++
        }
    }

    val floats =
        Array(labelCount) { FloatArray(numClasses) }

    for (i in 0 until labelCount) {
        floats[i] =
            Dataset.toOneHotVector(
                numClasses,
                labelBuffer[i]
            )
    }
    return floats
}