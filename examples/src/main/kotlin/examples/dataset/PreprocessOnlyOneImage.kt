/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import org.jetbrains.kotlinx.dl.visualization.swing.ImagesJPanel3
import java.awt.Color
import java.awt.Graphics
import java.io.File
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min

/**
 * This example shows how to do image preprocessing using [Preprocessing] for only one image.
 *
 * Also we use the [JPanel] to visualise (a back part of the pigeon should be displayed).
 *
 * It includes:
 * - image preprocessing
 * - image visualisation
 */
fun main() {
    val image =
        File("C:\\Users\\zaleslaw\\IdeaProjects\\KotlinDL\\examples\\src\\main\\resources\\datasets\\vgg\\image2.jpg")

    val preprocessedImagesDirectory =
        File("C:\\Users\\zaleslaw\\processedImages")

    val preprocessing: Preprocessing = preprocess {
        transformImage {
            load {
                pathToData = image
                imageShape = ImageShape(224, 224, 3)
                colorMode = ColorOrder.BGR
            }
            rotate {
                degrees = 0f
            }
            crop {
                left = 100
                right = 0
                top = 100
                bottom = 0
            }
            resize {
                outputWidth = 400
                outputHeight = 400
                interpolation = InterpolationType.NEAREST
            }
            save {
                dirLocation = preprocessedImagesDirectory
            }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val rawImage = preprocessing().first

    val frame = JFrame("Filters")
    frame.contentPane.add(ImagesJPanel3(rawImage, ImageShape(400, 400, 3)))
    frame.setSize(1000, 1000)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}



