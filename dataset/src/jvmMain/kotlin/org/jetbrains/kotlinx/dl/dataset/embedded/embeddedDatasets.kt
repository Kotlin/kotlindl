/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.embedded

import io.jhdf.HdfFile
import io.jhdf.api.Dataset
import org.jetbrains.kotlinx.dl.api.inference.loaders.AWS_S3_URL
import org.jetbrains.kotlinx.dl.api.inference.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.audio.wav.WavFile
import java.io.*
import java.net.URL
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import java.nio.file.StandardCopyOption
import java.util.zip.ZipEntry
import java.util.zip.ZipFile


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
    cacheDirectory.existsOrMkdirs()

    val trainXpath = loadFile(cacheDirectory, TRAIN_IMAGES_ARCHIVE).absolutePath
    val trainYpath = loadFile(cacheDirectory, TRAIN_LABELS_ARCHIVE).absolutePath
    val testXpath = loadFile(cacheDirectory, TEST_IMAGES_ARCHIVE).absolutePath
    val testYpath = loadFile(cacheDirectory, TEST_LABELS_ARCHIVE).absolutePath

    return OnHeapDataset.createTrainAndTestDatasets(
        trainXpath,
        trainYpath,
        testXpath,
        testYpath,
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
    cacheDirectory.existsOrMkdirs()

    val trainXpath = loadFile(cacheDirectory, FASHION_TRAIN_IMAGES_ARCHIVE).absolutePath
    val trainYpath = loadFile(cacheDirectory, FASHION_TRAIN_LABELS_ARCHIVE).absolutePath
    val testXpath = loadFile(cacheDirectory, FASHION_TEST_IMAGES_ARCHIVE).absolutePath
    val testYpath = loadFile(cacheDirectory, FASHION_TEST_LABELS_ARCHIVE).absolutePath

    return OnHeapDataset.createTrainAndTestDatasets(
        trainXpath,
        trainYpath,
        testXpath,
        testYpath,
        ::extractImages,
        ::extractLabels
    )
}

/** Path to H5 file of Mnist 3D Dataset. */
public const val MNIST_3D_DATASET: String = "datasets/mnist-3d/dataset.h5"

/**
 * Loads the [MNIST 3D dataset](https://www.kaggle.com/daavoo/3d-mnist).
 * This is a dataset of 10,000 16x16x16 grayscale 3D images of the 10 digits,
 * along with a test set of 2,000 3D images.
 *
 * NOTE: Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset,
 * which is a derivative work from original NIST datasets.
 * MNIST dataset is made available under the terms of the
 * [Creative Commons Attribution-Share Alike 3.0 license.](https://creativecommons.org/licenses/by-sa/3.0/)
 * MNIST 3D dataset was created by [daavoo](https://github.com/daavoo) as a transformation of
 * original MNIST dataset to 3D images to provide simple example of working with 3D images.
 *
 * @param [cacheDirectory] Cache directory to cached models and datasets.
 *
 * @return Train and test datasets. Each dataset includes X and Y data.
 * X data are float arrays of grayscale image data with shapes (num_samples, 16, 16, 16).
 * Y data float arrays of digit labels (integers in range 0-9) with shapes (num_samples,).
 */
public fun mnist3D(cacheDirectory: File = File("cache")): Pair<OnHeapDataset, OnHeapDataset> {
    cacheDirectory.existsOrMkdirs()

    return HdfFile(loadFile(cacheDirectory, MNIST_3D_DATASET)).use {

        val (trainData, trainLabels) = it.extractMnist3DDataset("train")
        val (testData, testLabels) = it.extractMnist3DDataset("test")

        Pair(
            OnHeapDataset.create(trainData, trainLabels),
            OnHeapDataset.create(testData, testLabels)
        )
    }
}

/** Extract mnist3d X data from HD5 file [dataset] */
private fun extractMnist3DData(dataset: Dataset) =
    (dataset.data as Array<*>)
        .map { (it as DoubleArray).map(Double::toFloat).toFloatArray() }
        .toTypedArray()

/** Extract mnist3d Y labels from HD5 file [dataset] */
private fun extractMnist3DLabels(dataset: Dataset) =
    (dataset.data as LongArray).map(Long::toFloat).toFloatArray()

/** Extract mnist3d data and labels from HD5 file under specified [label] */
private fun HdfFile.extractMnist3DDataset(label: String): Pair<Array<FloatArray>, FloatArray> =
    Pair(
        extractMnist3DData(getDatasetByPath("X_$label")),
        extractMnist3DLabels(getDatasetByPath("y_$label"))
    )

/** Data size of [Free Spoken Digits Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset). */
public const val FSDD_SOUND_DATA_SIZE: Long = 20480

/**
 * Loads the [Free Spoken Digits Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
 * This is a dataset of wav sound files of the 10 digits spoken by different people many times each.
 * The test set officially consists of the first 10% of the recordings. Recordings numbered 0-4 (inclusive)
 * are in the test and 5-49 are in the training set.
 *
 * As the input data files have different number of channels of data we split every input file into separate samples
 * that are threatened as separate samples with the same label.
 *
 * Free Spoken Digits Dataset is made available under the terms of the
 * [Creative Commons Attribution-ShareAlike 4.0 International.](https://creativecommons.org/licenses/by-sa/4.0/)
 *
 * @param [cacheDirectory] Cache directory to cached models and datasets.
 * @param [maxTestIndex] Index of max sample to be selected to test part of data.
 *
 * @return Train and test datasets. Each dataset includes X and Y data. X data are float arrays of sound data with shapes
 * (num_samples, FSDD_SOUND_DATA_SIZE) where FSDD_SOUND_DATA_SIZE is at least as long as the longest input sequence and all
 * sequences are padded with zeros to have equal length. Y data float arrays of digit labels (integers in range 0-9)
 * with shapes (num_samples,).
 */
public fun freeSpokenDigits(
    cacheDirectory: File = File("cache"),
    maxTestIndex: Int = 5
): Pair<OnHeapDataset, OnHeapDataset> {
    cacheDirectory.existsOrMkdirs()

    val path = freeSpokenDigitDatasetPath(cacheDirectory)
    val dataset = File(path)
        .listFiles()?.flatMap(::extractWavFileSamples)
        ?: throw IllegalStateException("Cannot find Free Spoken Digits Dataset files in $path")
    val maxDataSize = dataset.maxOfOrNull { it.first.size }
        ?: throw IllegalStateException("Empty Free Spoken Digits Dataset")
    check(maxDataSize <= FSDD_SOUND_DATA_SIZE) {
        "Sound data should be limited to $FSDD_SOUND_DATA_SIZE values but has $maxDataSize"
    }
    val data = dataset.map(::extractPaddedDataWithIndex)
    val labels = dataset.map(::extractLabelWithIndex)

    val (trainData, testData) = data.splitToTrainAndTestByIndex(maxTestIndex)
    val (trainLabels, testLabels) = labels.splitToTrainAndTestByIndex(maxTestIndex)

    return Pair(
        OnHeapDataset.create(trainData, trainLabels.toFloatArray()),
        OnHeapDataset.create(testData, testLabels.toFloatArray())
    )
}

/**
 * Extract wav file samples from given file and return a list of data from all its
 * channels as a triple of (channel_data, label, sample_index)
 *
 * @param file to read from the sound data
 * @return list of triples (channel_data, label, sample_index) from all channels from file
 */
private fun extractWavFileSamples(file: File): List<Triple<FloatArray, Float, Int>> =
    WavFile(file).use {
        val data = it.readRemainingFrames()
        val parts = file.name.split("_")
        val label = parts[0].toFloat()
        val index = parts[2].split(".")[0].toInt()
        data.map { channel -> Triple(channel, label, index) }
    }

private fun extractPaddedDataWithIndex(dataLabelIndex: Triple<FloatArray, Float, Int>): Pair<FloatArray, Int> =
    Pair(dataLabelIndex.first.copyInto(FloatArray(FSDD_SOUND_DATA_SIZE.toInt())), dataLabelIndex.third)

private fun extractLabelWithIndex(dataLabelIndex: Triple<FloatArray, Float, Int>): Pair<Float, Int> =
    Pair(dataLabelIndex.second, dataLabelIndex.third)

private inline fun <reified T> List<Pair<T, Int>>.splitToTrainAndTestByIndex(maxTestIndex: Int): Pair<Array<T>, Array<T>> {
    val test = filter { it.second < maxTestIndex }.map { it.first }.toTypedArray()
    val train = filter { it.second >= maxTestIndex }.map { it.first }.toTypedArray()
    return Pair(train, test)
}

/** Path to train images archive of Mnist Dataset. */
private const val CIFAR_10_IMAGES_ARCHIVE: String = "datasets/cifar10/images.zip"

/** Path to train labels archive of Mnist Dataset. */
private const val CIFAR_10_LABELS_ARCHIVE: String = "datasets/cifar10/trainLabels.csv"

/** Returns paths to images and its labels for the Cifar'10 dataset. */
public fun cifar10Paths(cacheDirectory: File = File("cache")): Pair<String, String> {
    cacheDirectory.existsOrMkdirs()

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
private const val DOGS_CATS_IMAGES_ARCHIVE: String = "datasets/dogs-vs-cats/data.zip"

/** Returns path to images of the Dogs-vs-Cats dataset. */
public fun dogsCatsDatasetPath(cacheDirectory: File = File("cache")): String =
    unzipDatasetPath(
        cacheDirectory,
        loadFile(cacheDirectory, DOGS_CATS_IMAGES_ARCHIVE),
        "/datasets/dogs-vs-cats"
    )

/** Path to the subset of Dogs-vs-Cats dataset. */
private const val DOGS_CATS_SMALL_IMAGES_ARCHIVE: String = "datasets/small-dogs-vs-cats/data.zip"

/** Returns path to images of the subset of the Dogs-vs-Cats dataset. */
public fun dogsCatsSmallDatasetPath(cacheDirectory: File = File("cache")): String =
    unzipDatasetPath(
        cacheDirectory,
        loadFile(cacheDirectory, DOGS_CATS_SMALL_IMAGES_ARCHIVE),
        "/datasets/small-dogs-vs-cats"
    )

/** Path to the Free Spoken Digits Dataset. */
private const val FSDD_SOUNDS_ARCHIVE: String = "datasets/fsdd.zip"

/** Path to download the Free Spoken Digits Dataset. */
private const val FSS_SOUNDS_SOURCE: String =
    "https://codeload.github.com/Jakobovski/free-spoken-digit-dataset/zip/refs/heads/master"

/** Returns path to sound data files from Free Spoken Digits Dataset. */
public fun freeSpokenDigitDatasetPath(cacheDirectory: File = File("cache")): String =
    unzipDatasetPath(
        cacheDirectory,
        loadFile(cacheDirectory, FSDD_SOUNDS_ARCHIVE, downloadURLFromRelativePath = { FSS_SOUNDS_SOURCE }),
        "/datasets/free-spoken-digit"
    ).run {
        "$this/free-spoken-digit-dataset-master/recordings"
    }

/**
 * Download the compressed dataset from external source, decompress the file and remove the downloaded file
 * but leave the decompressed data from dataset.
 *
 * @param [cacheDirectory] The directory where the downloaded files are stored.
 * @param [archive] Archive file.
 * @param [dirRelativePath] The relative path where to store the downloaded archive temporarily and decompress its data.
 * @return The absolute path string to directory where dataset is decompressed.
 */
private fun unzipDatasetPath(cacheDirectory: File, archive: File, dirRelativePath: String): String {
    cacheDirectory.existsOrMkdirs()

    val dataDirectory = File(cacheDirectory.absolutePath + dirRelativePath)
    val toFolder = dataDirectory.toPath()

    if (!dataDirectory.exists()) Files.createDirectories(dataDirectory.toPath())

    if (archive.exists()) {
        extractFromZipArchiveToFolder(archive.toPath(), toFolder)
        val deleted = archive.delete()
        if (!deleted) {
            throw Exception("Archive ${archive.absolutePath} could not be deleted! Delete this archive manually.")
        }
    } else {
        throw Exception("No archive file ${archive.absolutePath} in the cache folder!")
    }

    return toFolder.toAbsolutePath().toString()
}

/**
 * Downloads a file from a URL if it not already in the cache.
 *
 * By default, the download location
 * is defined as the concatenation of [AWS_S3_URL] and [relativePathToFile] but can be defined
 * as an arbitrary file location to download file from *
 *
 * @param [cacheDirectory] where the downloaded file is stored
 * @param [relativePathToFile] where the downloaded file is stored in [cacheDirectory] and which can
 * define the location of file to be downloaded
 * @param [downloadURLFromRelativePath] can produce the download URL of the file using. Defaults to [AWS_S3_URL]/[relativePathToFile].
 * @param [loadingMode] of the file to be loaded. Defaults to [LoadingMode.SKIP_LOADING_IF_EXISTS]
 * @return downloaded [File] on local file system.
 */
private fun loadFile(
    cacheDirectory: File,
    relativePathToFile: String,
    downloadURLFromRelativePath: (String) -> String = { "$AWS_S3_URL/$it" },
    loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS
): File {
    val fileName = cacheDirectory.absolutePath + "/" + relativePathToFile
    val file = File(fileName)
    file.parentFile.mkdirs() // Will create parent directories if not exists

    if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
        val urlString = downloadURLFromRelativePath(relativePathToFile)
        val inputStream = URL(urlString).openStream()
        Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
    }

    return file
}

/** Creates file structure archived in zip file with all directories and subdirectories. */
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

internal fun File.existsOrMkdirs() {
    if (!exists()) {
        val created = mkdirs()
        if (!created) {
            throw Exception("Directory $absolutePath could not be created! Create this directory manually.")
        }
    }
}
