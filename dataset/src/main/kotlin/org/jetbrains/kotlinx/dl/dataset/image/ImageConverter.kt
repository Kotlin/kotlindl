/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.image

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import java.awt.image.BufferedImage
import java.awt.image.BufferedImage.TYPE_3BYTE_BGR
import java.awt.image.DataBufferByte
import java.io.File
import java.io.IOException
import java.io.InputStream
import javax.imageio.ImageIO


/** Helper object to convert images to [FloatArray]. */
public object ImageConverter {
    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(image: BufferedImage, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return OnHeapDataset.toRawVector(imageToByteArray(image, colorOrder))
    }

    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(inputStream: InputStream, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return toRawFloatArray(toBufferedImage(inputStream), colorOrder)
    }

    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return imageFile.inputStream().use { toRawFloatArray(it, colorOrder) }
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(image: BufferedImage, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return OnHeapDataset.toNormalizedVector(imageToByteArray(image, colorOrder))
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(inputStream: InputStream, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return toNormalizedFloatArray(toBufferedImage(inputStream), colorOrder)
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
        return imageFile.inputStream().use { toNormalizedFloatArray(it, colorOrder) }
    }

    /**
     * Returns [BufferedImage] extracted from [inputStream].
     */
    @Throws(IOException::class)
    public fun toBufferedImage(inputStream: InputStream): BufferedImage {
        ImageIO.setUseCache(false)
        return ImageIO.read(inputStream)
    }

    private fun imageToByteArray(image: BufferedImage, colorOrder: ColorOrder): ByteArray {
        var res = (image.raster.dataBuffer as DataBufferByte).data // pixels
        check(image.alphaRaster == null) { "Images with alpha channels are not supported yet!" }
        check(image.type == TYPE_3BYTE_BGR) { "Images with image type (constant from BufferedImage class) ${image.type} are not supported!" }

        if (image.type == TYPE_3BYTE_BGR && colorOrder == ColorOrder.RGB) {
            res = swapRandB(res)
        }
        return res
    }

    private fun swapRandB(res: ByteArray): ByteArray {
        for (i in res.indices) {
            if (i % 3 == 2) { // swap i and i-2 elements from BGR to RGB
                val tmp = res[i]
                res[i] = res[i - 2]
                res[i - 2] = tmp
            }
        }
        return res
    }

    /** Converts [image] with [colorOrder] to the 3D array. */
    public fun imageTo3DFloatArray(
        image: BufferedImage,
        colorOrder: ColorOrder = ColorOrder.BGR
    ): Array<Array<FloatArray>> {
        val pixels = (image.raster.dataBuffer as DataBufferByte).data
        val width = image.width
        val height = image.height
        val hasAlphaChannel = image.alphaRaster != null
        val lastDimensions = if (hasAlphaChannel) 4 else 3

        val result = Array(height) { Array(width) { FloatArray(lastDimensions) } }
        if (hasAlphaChannel) {
            val pixelLength = 4
            var pixel = 0
            var row = 0
            var col = 0
            while (pixel < pixels.size) {
                result[row][col][3] = (pixels[pixel].toInt() and 0xff shl 24).toFloat() // alpha
                result[row][col][0] = (pixels[pixel + 1].toInt() and 0xff).toFloat() // blue
                if (colorOrder == ColorOrder.RGB) {
                    result[row][col][1] = (pixels[pixel + 2].toInt() and 0xff shl 8).toFloat() // green
                    result[row][col][2] = (pixels[pixel + 3].toInt() and 0xff shl 16).toFloat() // red
                } else {
                    result[row][col][2] = (pixels[pixel + 2].toInt() and 0xff shl 8).toFloat() // green
                    result[row][col][1] = (pixels[pixel + 3].toInt() and 0xff shl 16).toFloat() // red
                }

                col++
                if (col == width) {
                    col = 0
                    row++
                }
                pixel += pixelLength
            }
        } else {
            val pixelLength = 3
            var pixel = 0
            var row = 0
            var col = 0
            while (pixel < pixels.size) {
                result[row][col][0] = (pixels[pixel].toInt() and 0xff).toFloat() // blue
                if (colorOrder == ColorOrder.RGB) {
                    result[row][col][1] = (pixels[pixel + 1].toInt() and 0xff shl 8).toFloat() // green
                    result[row][col][2] = (pixels[pixel + 2].toInt() and 0xff shl 16).toFloat() // red
                } else {
                    result[row][col][2] = (pixels[pixel + 1].toInt() and 0xff shl 8).toFloat() // green
                    result[row][col][1] = (pixels[pixel + 2].toInt() and 0xff shl 16).toFloat() // red
                }
                col++
                if (col == width) {
                    col = 0
                    row++
                }
                pixel += pixelLength
            }
        }
        return result
    }
}

/** Represents the order of colors in pixel reading. */
public enum class ColorOrder {
    /** Red, green, blue. */
    RGB,

    /** Blue, green, red. */
    BGR
}
