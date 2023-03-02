/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.generator.EmptyLabels
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.toFloatArray
import org.jetbrains.kotlinx.dl.impl.preprocessing.mean
import org.jetbrains.kotlinx.dl.impl.preprocessing.normalize
import org.jetbrains.kotlinx.dl.impl.preprocessing.std
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Paths

/**
 * This example demonstrates [normalize] tensor preprocessor.
 * It shows how to compute mean and std values for the dataset and how to use these values for normalization.
 */
fun main() {
    val resource = Operation::class.java.getResource("/datasets/vgg")!!
    val imageDirectory = Paths.get(resource.toURI()).toFile()
    val images = OnHeapDataset.create(imageDirectory, EmptyLabels()).x
    val datasetMean = mean(*images, channels = 3)
    val datasetStd = std(*images, channels = 3)
    println("Dataset mean is ${datasetMean.contentToString()}\nDataset std is ${datasetStd.contentToString()}")

    val imageResource = Operation::class.java.getResource("/datasets/vgg/image2.jpg")
    val image = File(imageResource!!.toURI())
    val imageFloats = ImageConverter.toRawFloatArray(image)
    println(
        "Raw image mean is ${imageFloats.mean(3).contentToString()}\n" +
                "Raw image std is ${imageFloats.std(3).contentToString()}"
    )

    val preprocessing = pipeline<BufferedImage>()
        .toFloatArray { }
        .normalize {
            mean = datasetMean
            std = datasetStd
        }

    val (processedImageFloats, _) = preprocessing.fileLoader().load(image)

    println(
        "Processed image mean is ${processedImageFloats.mean(3).contentToString()}\n" +
                "Processed image std is ${processedImageFloats.std(3).contentToString()}"
    )
}