/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.AWS_S3_URL
import org.jetbrains.kotlinx.dl.api.inference.keras.loaders.LoadingMode
import org.jetbrains.kotlinx.dl.dataset.handler.*
import java.io.File
import java.net.URL
import java.nio.file.Files
import java.nio.file.Paths
import java.nio.file.StandardCopyOption

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
 * |   1   | Trouser     |
 * |   2   | Pullover    |
 * |   3   | Dress       |
 * |   4   | Coat        |
 * |   5   | Sandal      |
 * |   6   | Shirt       |
 * |   7   | Sneaker     |
 * |   8   | Bag         |
 * |   9   | Ankle boot  |
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

/** Downloads a file from a URL if it not already in the cache. */
private fun loadFile(
    cacheDirectory: File,
    relativePathToFile: String,
    loadingMode: LoadingMode = LoadingMode.SKIP_LOADING_IF_EXISTS
): File {
    val fileName = cacheDirectory.absolutePath + "/" + relativePathToFile
    val urlString = "$AWS_S3_URL/$relativePathToFile"
    val file = File(fileName)

    file.parentFile.mkdirs(); // Will create parent directories if not exists

    if (!file.exists() || loadingMode == LoadingMode.OVERRIDE_IF_EXISTS) {
        val inputStream = URL(urlString).openStream()
        Files.copy(inputStream, Paths.get(fileName), StandardCopyOption.REPLACE_EXISTING)
    }

    return File(fileName)
}
