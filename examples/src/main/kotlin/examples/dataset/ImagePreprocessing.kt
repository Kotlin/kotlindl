/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.generator.EmptyLabels
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.rescale
import org.jetbrains.kotlinx.dl.visualization.swing.ImagePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.image.BufferedImage
import java.net.URL
import java.nio.file.Paths

/**
 * This example shows how to do image preprocessing from scratch using preprocessing DSL.
 *
 * It includes:
 * - dataset creation from images located in resource folder;
 * - image preprocessing;
 * - image visualisation with the [ImagePanel].
 */
fun main() {
    val preprocessing = pipeline<BufferedImage>()
        .crop {
            left = 12
            right = 12
            top = 12
            bottom = 12
        }
        .rotate {
            degrees = 60f
        }
        .resize {
            outputWidth = 300
            outputHeight = 300
            interpolation = InterpolationType.NEAREST
        }
        .grayscale()
        .toFloatArray { }
        .rescale {
            scalingCoefficient = 255f
        }

    val resource: URL = Operation::class.java.getResource("/datasets/vgg")
    val imageDirectory = Paths.get(resource.toURI()).toFile()
    val dataset = OnFlyImageDataset.create(imageDirectory, EmptyLabels(), preprocessing)
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(8)

    val rawImage = batchIter.next().x[2]

    val image = ImageConverter.floatArrayToBufferedImage(
        rawImage,
        preprocessing.getOutputShape(TensorShape(-1, -1, 3)),
        ColorMode.GRAYSCALE,
        isNormalized = true
    )

    showFrame("Filters", ImagePanel(image))
}
