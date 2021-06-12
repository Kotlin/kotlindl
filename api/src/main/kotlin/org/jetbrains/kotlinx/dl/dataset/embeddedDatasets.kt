/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.AWS_S3_URL
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.dataset.handler.*
import org.jetbrains.kotlinx.dl.dataset.sound.wav.WavFile
import java.io.*
import java.net.URL
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import java.io.File

import java.io.IOException

import java.io.FileOutputStream
import java.lang.IllegalStateException


/**
 * Loads the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
 * This is a dataset of 60,000 28x28 grayscale images of the 10 digits,
 * along with a test set of 10,000 images.
 * More info can be found at the [MNIST homepage](http://yann.lecun.com/exdb/mnist/).
 *
 * NOTE: Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
 * which is a derivative work from original NIST datasets.
 * MNIST dataset is made available under the terms of the
 * [Creative Commons Attribution-Share Alike 3.0 license.](https://creativecommons.org/licenses/by-sa/3.0/)
 *
 * @param [cacheDirectory] Cache directory to cached models and datasets.
 *
 * @return Train and test datasets. Each dataset includes X and Y data. X data are uint8 arrays of grayscale image data with shapes
 * (num_samples, 28, 28). Y data uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
 */
public fun mnist(cacheDirectory: File = File("cache")): Pair<OnHeapDataset, OnHeapDataset> {
    if (!cacheDirectory.exists()) {
        val created = cacheDirectory.mkdir()
        if (!created) throw Exception("Directory ${cacheDirectory.absolutePath} could not be created! Create this directory manually.")
    }

    val trainXpath = loadFile(cacheDirectory, TRAIN_IMAGES_ARCHIVE).absolutePath
    val trainYpath = loadFile(cacheDirectory, TRAIN_LABELS_ARCHIVE).absolutePath
    val testXpath = loadFile(cacheDirectory, TEST_IMAGES_ARCHIVE).absolutePath
    val testYpath = loadFile(cacheDirectory, TEST_LABELS_ARCHIVE).absolutePath

    return OnHeapDataset.createTrainAndTestDatasets(
        trainXpath,
        trainYpath,
        testXpath,
        testYpath,
        NUMBER_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )
}

/**
 * Loads the Fashion-MNIST dataset.
 *
 * This is a dataset of 60,000 28x28 grayscale images of 10 fashion categories,
 * along with a test set of 10,000 images. This dataset can be used as
 * a drop-in replacement for MNIST. The class labels are:
 *
 * | Label | Description |
 * |:-----:|-------------|
 * |   0   | T-shirt/top |
 * |   1   | Trousers     |
 * |   2   | Pullover    |
 * |   3   | Dress       |
 * |   4   | Coat        |
 * |   5   | Sandals      |
 * |   6   | Shirt       |
 * |   7   | Sneakers     |
 * |   8   | Bag         |
 * |   9   | Ankle boots  |
 *
 * NOTE: The copyright for Fashion-MNIST is held by Zalando SE.
 * Fashion-MNIST is licensed under the [MIT license](https://github.com/zalandoresearch/fashion-mnist/blob/master/LICENSE).
 *
 * @param [cacheDirectory] Cache directory to cached models and datasets.
 *
 * @return Train and test datasets. Each dataset includes X and Y data. X data are uint8 arrays of grayscale image data with shapes
 * (num_samples, 28, 28). Y data uint8 arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
 */
public fun fashionMnist(cacheDirectory: File = File("cache")): Pair<OnHeapDataset, OnHeapDataset> {
    if (!cacheDirectory.exists()) {
        val created = cacheDirectory.mkdir()
        if (!created) throw Exception("Directory ${cacheDirectory.absolutePath} could not be created! Create this directory manually.")
    }

    val trainXpath = loadFile(cacheDirectory, FASHION_TRAIN_IMAGES_ARCHIVE).absolutePath
    val trainYpath = loadFile(cacheDirectory, FASHION_TRAIN_LABELS_ARCHIVE).absolutePath
    val testXpath = loadFile(cacheDirectory, FASHION_TEST_IMAGES_ARCHIVE).absolutePath
    val testYpath = loadFile(cacheDirectory, FASHION_TEST_LABELS_ARCHIVE).absolutePath

    return OnHeapDataset.createTrainAndTestDatasets(
        trainXpath,
        trainYpath,
        testXpath,
        testYpath,
        NUMBER_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )
}

public const val FSDD_SOUND_DATA_SIZE: Long = 20480

public fun freeSpokenDigits(
    cacheDirectory: File = File("cache")
): OnHeapDataset {
    val path = freeSpokenDigitDatasetPath(cacheDirectory)
    val dataset = File(path).listFiles()?.flatMap(::extractWavFileSamples)
        ?: throw IllegalStateException("Cannot find Free Spoken Digits Dataset files in $path")
    val maxDataSize = dataset.map { it.first.size }.maxOrNull()
        ?: throw IllegalStateException("Empty Free Spoken Digits Dataset")
    require(maxDataSize <= FSDD_SOUND_DATA_SIZE) {
        "Sound data should be limited to $FSDD_SOUND_DATA_SIZE values but has $maxDataSize"
    }
    val data = dataset.map { it.first.copyInto(FloatArray(FSDD_SOUND_DATA_SIZE.toInt())) }.toTypedArray()
    val labels = dataset.map { it.second }.toTypedArray().toFloatArray()
    return OnHeapDataset(data, labels).shuffle()
}

private fun extractWavFileSamples(file: File): List<Pair<FloatArray, Float>> =
    WavFile(file).use {
        val data = it.readRemainingFrames()
        val label = file.name.split("_")[0].toFloat()
        data.map { channel -> Pair(channel, label) }
    }

/** Path to train images archive of Mnist Dataset. */
private const val CIFAR_10_IMAGES_ARCHIVE: String = "datasets/cifar10/images.zip"

/** Path to train labels archive of Mnist Dataset. */
private const val CIFAR_10_LABELS_ARCHIVE: String = "datasets/cifar10/trainLabels.csv"

/** Returns paths to images and its labels for the Cifar'10 dataset. */
public fun cifar10Paths(cacheDirectory: File = File("cache")): Pair<String, String> {
    if (!cacheDirectory.exists()) {
        val created = cacheDirectory.mkdir()
        if (!created) throw Exception("Directory ${cacheDirectory.absolutePath} could not be created! Create this directory manually.")
    }

    val pathToLabel = loadFile(cacheDirectory, CIFAR_10_LABELS_ARCHIVE).absolutePath

    val datasetDirectory = File(cacheDirectory.absolutePath + "/datasets/cifar10")
    val toFolder = datasetDirectory.toPath()

    val imageDataDirectory = File(cacheDirectory.absolutePath + "/datasets/cifar10/images")
    if (!imageDataDirectory.exists()) {
        Files.createDirectories(imageDataDirectory.toPath())

        val pathToImageArchive = loadFile(cacheDirectory, CIFAR_10_IMAGES_ARCHIVE)
        extractFromZipArchiveToFolder(pathToImageArchive.toPath(), toFolder)
        val deleted = pathToImageArchive.delete()
        if (!deleted) throw Exception("Archive ${pathToImageArchive.absolutePath} could not be deleted! Create this archive manually.")
    }

    return Pair(imageDataDirectory.toPath().toAbsolutePath().toString(), pathToLabel)
}

/** Path to the Dogs-vs-Cats dataset. */
private const val DOGS_CATS_IMAGES_ARCHIVE: String = "datasets/catdogs/data.zip"

/** Returns path to images of the Dogs-vs-Cats dataset. */
public fun dogsCatsDatasetPath(cacheDirectory: File = File("cache")): String =
    datasetPath(cacheDirectory, DOGS_CATS_IMAGES_ARCHIVE, "/datasets/dogs-vs-cats")

/** Path to the subset of Dogs-vs-Cats dataset. */
private const val DOGS_CATS_SMALL_IMAGES_ARCHIVE: String = "datasets/small_catdogs/data.zip"

/** Returns path to images of the subset of the Dogs-vs-Cats dataset. */
public fun dogsCatsSmallDatasetPath(cacheDirectory: File = File("cache")): String =
    datasetPath(cacheDirectory, DOGS_CATS_SMALL_IMAGES_ARCHIVE, "/datasets/small-dogs-vs-cats")

/** Path to the subset of Dogs-vs-Cats dataset. */
private const val FSDD_SOUNDS_ARCHIVE: String = "datasets/fsdd/data.zip"

/** Returns path to images of the subset of the Dogs-vs-Cats dataset. */
public fun freeSpokenDigitDatasetPath(cacheDirectory: File = File("cache")): String =
    datasetPath(cacheDirectory, FSDD_SOUNDS_ARCHIVE, "/datasets/free-spoken-digit")

private fun datasetPath(cacheDirectory: File, archiveRelativePath: String, dirRelativePath: String): String {
    if (!cacheDirectory.exists()) {
        val created = cacheDirectory.mkdir()
        if (!created) {
            throw Exception("Directory ${cacheDirectory.absolutePath} could not be created! Create this directory manually.")
        }
    }

    val dataDirectory = File(cacheDirectory.absolutePath + dirRelativePath)
    val toFolder = dataDirectory.toPath()

    if (!dataDirectory.exists()) {
        Files.createDirectories(dataDirectory.toPath())

        val pathToArchive = loadFile(cacheDirectory, archiveRelativePath)
        extractFromZipArchiveToFolder(pathToArchive.toPath(), toFolder)
        val deleted = pathToArchive.delete()
        if (!deleted) {
            throw Exception("Archive ${pathToArchive.absolutePath} could not be deleted! Create this archive manually.")
        }
    }

    return toFolder.toAbsolutePath().toString()
}

/**
 * Downloads a file from a URL if it not already in the cache. By default the download location
 * is defined as the concatenation of [AWS_S3_URL] and [relativePathToFile] but can be defined
 * as an arbitrary file location to download file from *
 *
 * @param cacheDirectory where the downloaded file is stored
 * @param relativePathToFile where the downloaded file is stored in [cacheDirectory] and which can
 * define the location of file to be downloaded
 * @param loadingMode of the file to be loaded. Defaults to [LoadingMode.SKIP_LOADING_IF_EXISTS]
 * @param urlLocation source of location of file to be downloaded based on the [relativePathToFile]]
 * @return downloaded [File] on local file system
 */
private fun loadFile(
    cacheDirectory: File,
    relativePathToFile: String,
    loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS,
    urlLocation: (String) -> String = { "$AWS_S3_URL/$it" }
): File {
    val fileName = cacheDirectory.absolutePath + "/" + relativePathToFile
    val file = File(fileName)
    file.parentFile.mkdirs() // Will create parent directories if not exists

    if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
        val urlString = urlLocation(relativePathToFile)
        val inputStream = URL(urlString).openStream()
        Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
    }

    return File(fileName)
}

/** Creates file structure archived in zip file with all directories and sub-directories */
@Throws(IOException::class)
internal fun extractFromZipArchiveToFolder(zipArchivePath: Path, toFolder: Path, bufferSize: Int = 4096) {
    val zipFile = ZipFile(zipArchivePath.toFile())
    val entries = zipFile.entries()

    while (entries.hasMoreElements()) {
        val entry = entries.nextElement() as ZipEntry
        var currentEntry = entry.name
        currentEntry = currentEntry.replace('\\', '/')

        val destFile = File(toFolder.toFile(), currentEntry)

        val destinationParent = destFile.parentFile
        destinationParent.mkdirs()

        if (!entry.isDirectory && !destFile.exists()) {
            val inputStream = BufferedInputStream(
                zipFile.getInputStream(entry)
            )
            var currentByte: Int
            // establish buffer for writing file
            val data = ByteArray(bufferSize)

            // write the current file to disk
            val fos = FileOutputStream(destFile)
            val dest = BufferedOutputStream(
                fos,
                bufferSize
            )

            // read and write until last byte is encountered
            while (inputStream.read(data, 0, bufferSize).also { currentByte = it } != -1) {
                dest.write(data, 0, currentByte)
            }
            dest.flush()
            dest.close()
            inputStream.close()
        }
    }
    zipFile.close()
}
