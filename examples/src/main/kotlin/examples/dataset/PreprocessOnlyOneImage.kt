/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.awt.Color
import java.io.File
import javax.swing.JFrame

/**
 * This example shows how to do image preprocessing using [Preprocessing] DSL for only one image.
 *
 * It includes:
 * - image preprocessing;
 * - image visualisation with the [ImagePanel].
 */
fun main() {
    val preprocessedImagesDirectory = File("processedImages")

    val preprocessing: Preprocessing = preprocess {
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
            pad {
                top = 10
                bottom = 40
                left = 10
                right = 10
                mode = PaddingMode.Fill(Color.WHITE)
            }
            convert { colorMode = ColorMode.BGR }
        }
        transformTensor {
            rescale {
                scalingCoefficient = 255f
            }
        }
    }

    val imageResource = ImagePreprocessing::class.java.getResource("/datasets/vgg/image2.jpg")
    val image = File(imageResource!!.toURI())
    val rawImage = preprocessing(image).first

    val bufferedImage = ImageConverter.floatArrayToBufferedImage(
        rawImage,
        preprocessing.getFinalShape(),
        ColorMode.BGR,
        isNormalized = true
    )

    val frame = JFrame("Filters")
    frame.contentPane.add(ImagePanel(bufferedImage))
    frame.pack()
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}
