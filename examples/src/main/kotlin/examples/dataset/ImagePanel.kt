/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.imageType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import javax.swing.JPanel

/**
 * A [JPanel] to display an image represented by a [FloatArray].
 *
 * @param [image]      an image represented by a [FloatArray] with values from 0 to 1.
 *                     Size of the array should be the same as [ImageShape.numberOfElements].
 * @param [imageShape] a shape of the image. Values for [ImageShape.width], [ImageShape.height] and [ImageShape.channels]
 *                     should be not `null`, and consistent with the size of the given array.
 * @param [colorMode]  a [ColorMode] for the image, containing the information about channels and their order.
 *                     [ColorMode.channels] should be the same as [ImageShape.channels].
 */
class ImagePanel(image: FloatArray, imageShape: ImageShape, colorMode: ColorMode) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape, colorMode)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val x = (size.width - bufferedImage.width) / 2
        val y = (size.height - bufferedImage.height) / 2
        graphics.drawImage(bufferedImage, x, y, null)
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

private fun FloatArray.toBufferedImage(imageShape: ImageShape, colorMode: ColorMode): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), colorMode.imageType())
    val rgbArray = copyOf().also {
        if (colorMode == ColorMode.BGR) ImageConverter.swapRandB(it)
    }
    rgbArray.forEachIndexed { index, value -> rgbArray[index] = value * 255f }
    result.raster.setPixels(0, 0, result.width, result.height, rgbArray)
    return result
}