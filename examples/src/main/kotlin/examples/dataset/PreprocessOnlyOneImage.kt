/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessing.fileLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.preprocessing.onResult
import org.jetbrains.kotlinx.dl.impl.preprocessing.rescale
import org.jetbrains.kotlinx.dl.visualization.swing.ImagePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import java.awt.Color
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

/**
 * This example shows how to do image preprocessing using preprocessing DSL for only one image.
 *
 * It includes:
 * - image preprocessing;
 * - image visualisation with the [ImagePanel].
 */
fun main() {
    val preprocessing = pipeline<BufferedImage>()
        .crop {
            left = 100
            right = 0
            top = 100
            bottom = 0
        }
        .rotate {
            degrees = 0f
        }
        .resize {
            outputWidth = 400
            outputHeight = 400
            interpolation = InterpolationType.NEAREST
        }
        .onResult { ImageIO.write(it, "jpg", File("image2.jpg")) }
        .pad {
            top = 10
            bottom = 40
            left = 10
            right = 10
            mode = PaddingMode.Fill(Color.WHITE)
        }
        .convert { colorMode = ColorMode.BGR }
        .toFloatArray { }
        .rescale {
            scalingCoefficient = 255f
        }

    val imageResource = Operation::class.java.getResource("/datasets/vgg/image2.jpg")
    val image = File(imageResource!!.toURI())
    val (rawImage, shape) = preprocessing.fileLoader().load(image)

    val bufferedImage = ImageConverter.floatArrayToBufferedImage(rawImage, shape, ColorMode.BGR, isNormalized = true)

    showFrame("Filters", ImagePanel(bufferedImage))
}
