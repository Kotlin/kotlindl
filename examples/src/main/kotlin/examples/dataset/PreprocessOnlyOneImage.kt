/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
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
    val imageResource = ImagePreprocessing::class.java.getResource("/datasets/vgg/image2.jpg")
    val image = File(imageResource!!.toURI())
    val preprocessedImagesDirectory = File("processedImages")

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


class ImagesJPanel3(
    val dst: FloatArray,
    val imageShape: ImageShape
) : JPanel() {
    override fun paint(graphics: Graphics) {
        for (i in 0 until imageShape.height!!.toInt()) { // rows
            for (j in 0 until imageShape.width!!.toInt()) { // columns
                val pixelWidth = 2
                val pixelHeight = 2

                // y = columnIndex
                // x = rowIndex
                val y = 100 + i * pixelWidth
                val x = 100 + j * pixelHeight

                val r =
                    dst.get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val g =
                    dst.get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val b =
                    dst.get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val r1 = (min(1.0f, max(r * 0.8f, 0.0f)) * 255).toInt()
                val g1 = (min(1.0f, max(g * 0.8f, 0.0f)) * 255).toInt()
                val b1 = (min(1.0f, max(b * 0.8f, 0.0f)) * 255).toInt()
                val color = Color(r, g, b)
                graphics.color = color
                graphics.fillRect(x, y, pixelWidth, pixelHeight)
                graphics.color = Color.BLACK
                graphics.drawRect(x, y, pixelWidth, pixelHeight)
            }
        }
    }
}
