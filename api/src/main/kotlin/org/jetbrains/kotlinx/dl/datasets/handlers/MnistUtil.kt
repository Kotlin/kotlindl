/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.handlers

import org.jetbrains.kotlinx.dl.datasets.Dataset
import java.io.DataInputStream
import java.util.zip.GZIPInputStream

private const val IMAGE_ARCHIVE_MAGIC = 2051
private const val LABEL_ARCHIVE_MAGIC = 2049

/** Path to train images archive of Mnist Dataset. */
public const val TRAIN_IMAGES_ARCHIVE: String = "datasets/mnist/train-images-idx3-ubyte.gz"

/** Path to train labels archive of Mnist Dataset. */
public const val TRAIN_LABELS_ARCHIVE: String = "datasets/mnist/train-labels-idx1-ubyte.gz"

/** Path to test images archive of Mnist Dataset. */
public const val TEST_IMAGES_ARCHIVE: String = "datasets/mnist/t10k-images-idx3-ubyte.gz"

/** Path to test labels archive of Mnist Dataset. */
public const val TEST_LABELS_ARCHIVE: String = "datasets/mnist/t10k-labels-idx1-ubyte.gz"

/** Number of classes. */
public const val NUMBER_OF_CLASSES: Int = 10

/**
 * Extracts Fashion Mnist images from [archivePath].
 */
public fun extractImages(archivePath: String): Array<FloatArray> {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            Dataset::class.java.classLoader.getResourceAsStream(archivePath)
        )
    )
    val magic = archiveStream.readInt()
    require(IMAGE_ARCHIVE_MAGIC == magic) { "\"$archivePath\" is not a valid image archive" }
    val imageCount = archiveStream.readInt()
    val imageRows = archiveStream.readInt()
    val imageCols = archiveStream.readInt()
    println(
        String.format(
            "Extracting %d images of %dx%d from %s",
            imageCount,
            imageRows,
            imageCols,
            archivePath
        )
    )
    val images =
        Array(imageCount) { FloatArray(imageRows * imageCols) }
    val imageBuffer = ByteArray(imageRows * imageCols)
    for (i in 0 until imageCount) {
        archiveStream.readFully(imageBuffer)
        images[i] =
            Dataset.toNormalizedVector(
                imageBuffer
            )
    }
    return images
}

/**
 * Extracts Fashion Mnist labels from [archivePath] with number of classes [numClasses].
 */
public fun extractLabels(archivePath: String, numClasses: Int): Array<FloatArray> {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            Dataset::class.java.classLoader.getResourceAsStream(archivePath)
        )
    )
    val magic = archiveStream.readInt()
    require(LABEL_ARCHIVE_MAGIC == magic) { "\"$archivePath\" is not a valid image archive" }
    val labelCount = archiveStream.readInt()
    println(String.format("Extracting %d labels from %s", labelCount, archivePath))
    val labelBuffer = ByteArray(labelCount)
    archiveStream.readFully(labelBuffer)
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