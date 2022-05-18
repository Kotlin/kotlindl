/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.EmptyLabels
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.net.URL
import java.nio.file.Paths
import javax.swing.JFrame

/**
 * This example shows how to do image preprocessing from scratch using [Preprocessing] DSL.
 *
 * It includes:
 * - dataset creation from images located in resource folder;
 * - image preprocessing;
 * - image visualisation with the [ImagePanel].
 */
fun main() {
    val preprocessing: Preprocessing = preprocess {
        transformImage {
            crop {
                left = 12
                right = 12
                top = 12
                bottom = 12
            }
            rotate {
                degrees = 60f
            }
            resize {
                outputWidth = 300
                outputHeight = 300
                interpolation = InterpolationType.NEAREST
            }
            grayscale()
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val resource: URL = ImagePreprocessing::class.java.getResource("/datasets/vgg")
    val imageDirectory = Paths.get(resource.toURI()).toFile()
    val dataset = OnFlyImageDataset.create(imageDirectory, EmptyLabels(), preprocessing)
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(8)

    val rawImage = batchIter.next().x[2]

    val frame = JFrame("Filters")

    val image = ImageConverter.floatArrayToBufferedImage(
        rawImage,
        preprocessing.getFinalShape(),
        ColorMode.GRAYSCALE,
        isNormalized = true
    )

    frame.contentPane.add(ImagePanel(image))
    frame.pack()
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}
