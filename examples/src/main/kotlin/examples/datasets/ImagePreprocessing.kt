/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.datasets

import examples.cnn.cifar10.extractCifar10LabelsAnsSort
import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.datasets.*
import org.jetbrains.kotlinx.dl.datasets.image.ColorOrder
import java.awt.Color
import java.awt.Graphics
import java.io.FileReader
import java.util.*
import javax.swing.JFrame
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min

fun main() {
    val properties = Properties()
    val reader = FileReader("data.properties")
    properties.load(reader)

    val cifarImagesArchive = properties["cifarImagesArchive"] as String
    val cifarLabelsArchive = properties["cifarLabelsArchive"] as String

    // TODO: standartize, center and normalize be careful in terms https://machinelearningmastery.com/how-to-normalize-center-and-standardize-images-with-the-imagedatagenerator-in-keras/
    val imageDirectory = "C:\\Users\\zaleslaw\\IdeaProjects\\KotlinDL\\examples\\src\\main\\resources\\datasets\\vgg"

    val preprocessing: Preprocessing = preprocessing {
        imagePreprocessing {
            load {
                dirLocation = imageDirectory
                imageShape = ImageShape(224, 224, 3)
                colorMode = ColorOrder.BGR
            }
            rotate {
                degrees = 30f
            }
            crop {
                left = 12
                right = 12
                top = 12
                bottom = 12
            }
            resize {
                outputWidth = 400
                outputHeight = 400
                interpolation = InterpolationType.NEAREST
            }
        }
        rescale {
            scalingCoefficient = 225f
        }
    }

    val y = extractCifar10LabelsAnsSort(cifarLabelsArchive, 10)
    val dataset = OnFlyImageDataset.create(preprocessing, y)
    val batchIter: Dataset.BatchIterator = dataset.batchIterator(
        8
    )

    val rawImage = batchIter.next().x[0]

    val frame = JFrame("Filters")
    frame.contentPane.add(ImagesJPanel(rawImage, ImageShape(400, 400, 3)))
    frame.setSize(1000, 1000)
    frame.isVisible = true
    frame.defaultCloseOperation = JFrame.EXIT_ON_CLOSE
    frame.isResizable = false
}


class ImagesJPanel(
    val dst: FloatArray,
    val imageShape: ImageShape
) : JPanel() {
    override fun paint(graphics: Graphics) {
        for (i in 0 until imageShape.width.toInt()) {
            for (j in 0 until imageShape.height.toInt()) {
                val pixelWidth = 2
                val pixelHeight = 2
                var x = 100 + i * pixelWidth
                val y = 100 + j * pixelHeight

                val r =
                    dst.get3D(i, j, 2, imageShape.width.toInt(), imageShape.channels.toInt())
                val g =
                    dst.get3D(i, j, 1, imageShape.width.toInt(), imageShape.channels.toInt())
                val b =
                    dst.get3D(i, j, 0, imageShape.width.toInt(), imageShape.channels.toInt())
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
