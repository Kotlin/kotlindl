/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.dataset.Dataset
import org.jetbrains.kotlinx.dl.dataset.OnFlyImageDataset
import org.jetbrains.kotlinx.dl.dataset.handler.extractCifar10LabelsAnsSort
import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.preprocessor.*
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.*
import java.awt.Color
import java.awt.Graphics
import java.io.FileReader
import java.net.URL
import java.nio.file.Paths
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min

/**
 * This example shows how to do image preprocessing from scratch using [Preprocessing].
 *
 * Also we use the [JPanel] to visualise (rotated pigeon should be displayed).
 *
 * It includes:
 * - dataset creation from images located in resource folder
 * - image preprocessing
 * - image visualisation
 */
fun main() {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val cifarLabelsArchive = properties["cifarLabelsArchive"] as String

    val resource: URL = ImagePreprocessing::class.java.getResource("/datasets/vgg")
    val imageDirectory = Paths.get(resource.toURI()).toFile()

    val preprocessing: Preprocessing = preprocess {
        transformImage {
            load {
                pathToData = imageDirectory
                imageShape = ImageShape(224, 224, 3)
                colorMode = ColorOrder.BGR
            }
            rotate {
                degrees = 60f
            }
            crop {
                left = 12
                right = 12
                top = 12
                bottom = 12
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

    val y = extractCifar10LabelsAnsSort(cifarLabelsArchive, 10)
    val dataset = OnFlyImageDataset.create(preprocessing, y)
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(
        8
    )

    val rawImage = batchIter.next().x[1]

    val frame = JFrame("Filters")
    frame.contentPane.add(ImagesJPanel(rawImage, ImageShape(300, 300, 3)))
    frame.setSize(1000, 1000)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}


private class ImagesJPanel(
    val dst: FloatArray,
    val imageShape: ImageShape
) : JPanel() {
    override fun paint(graphics: Graphics) {
        for (i in 0 until imageShape.height!!.toInt()) { // rows
            for (j in 0 until imageShape.width!!.toInt()) { // columns
                val pixelWidth = 2
                val pixelHeight = 2
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
                val color = Color(r1, g1, b1)
                graphics.color = color
                graphics.fillRect(x, y, pixelWidth, pixelHeight)
                graphics.color = Color.BLACK
                graphics.drawRect(x, y, pixelWidth, pixelHeight)
            }
        }
    }
}
