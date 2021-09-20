/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.io.File
import javax.swing.JFrame

/**
 * This example shows how to do image preprocessing using [Preprocessing] for only one image.
 *
 * Also we use the [ImagePanel] to visualise (a back part of the pigeon should be displayed).
 *
 * It includes:
 * - image preprocessing
 * - image visualisation
 */
fun main() {
    val imageResource = ImagePreprocessing::class.java.getResource("/datasets/vgg/image2.jpg")
    val image = File(imageResource!!.toURI())
    val preprocessedImagesDirectory = File("processedImages")

    val preprocessing: Preprocessing = preprocess {
        load {
            pathToData = image
            imageShape = ImageShape(224, 224, 3)
            colorMode = ColorOrder.BGR
        }
        transformImage {
            crop {
                left = 100
                right = 0
                top = 100
                bottom = 0
            }
            rotate {
                degrees = 0f
            }
            resize {
                outputWidth = 400
                outputHeight = 400
                interpolation = InterpolationType.NEAREST
                save {
                    dirLocation = preprocessedImagesDirectory
                }
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
    frame.contentPane.add(ImagePanel(rawImage, preprocessing.finalShape))
    frame.pack()
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}
