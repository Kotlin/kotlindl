/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.handler.extractCifar10LabelsAnsSort
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.EmptyLabels
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.net.URL
import java.nio.file.Paths
import javax.swing.JFrame

/**
 * This example shows how to do image preprocessing from scratch using [Preprocessing].
 *
 * Also we use the [ImagePanel] to visualise (rotated pigeon should be displayed).
 *
 * It includes:
 * - dataset creation from images located in resource folder
 * - image preprocessing
 * - image visualisation
 */
fun main() {
    val resource: URL = ImagePreprocessing::class.java.getResource("/datasets/vgg")
    val imageDirectory = Paths.get(resource.toURI()).toFile()

    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = imageDirectory
            imageShape = ImageShape(224, 224, 3)
            colorMode = ColorOrder.BGR
            labelGenerator = EmptyLabels()
        }
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
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val dataset = OnFlyImageDataset.create(preprocessing)
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(
        8
    )

    val rawImage = batchIter.next().x[2]

    val frame = JFrame("Filters")
    frame.contentPane.add(ImagePanel(rawImage, preprocessing.finalShape))
    frame.pack()
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}
