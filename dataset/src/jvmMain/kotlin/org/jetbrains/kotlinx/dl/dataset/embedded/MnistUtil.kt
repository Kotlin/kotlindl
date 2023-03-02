/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.embedded

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.impl.util.toNormalizedVector
import java.io.DataInputStream
import java.io.FileInputStream
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
 * Extracts (Fashion) Mnist images from [archivePath].
 */
public fun extractImages(archivePath: String): Array<FloatArray> {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            FileInputStream(archivePath)
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
    val imageBuffer = ByteArray(imageRows * imageCols)
    val images = Array(imageCount) {
        archiveStream.readFully(imageBuffer)
        toNormalizedVector(imageBuffer)
    }
    return images
}

/**
 * Extracts (Fashion) Mnist labels from [archivePath] with number of classes [numClasses].
 */
public fun extractLabels(archivePath: String): FloatArray {
    val archiveStream = DataInputStream(
        GZIPInputStream(
            FileInputStream(archivePath)
        )
    )
    val magic = archiveStream.readInt()
    require(LABEL_ARCHIVE_MAGIC == magic) { "\"$archivePath\" is not a valid image archive" }
    val labelCount = archiveStream.readInt()
    println(String.format("Extracting %d labels from %s", labelCount, archivePath))
    val labelBuffer = ByteArray(labelCount)
    archiveStream.readFully(labelBuffer)
    val floats = FloatArray(labelCount)
    for (i in 0 until labelCount) {
        floats[i] =
            OnHeapDataset.convertByteToFloat(
                labelBuffer[i]
            )
    }
    return floats
}
