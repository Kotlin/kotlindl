/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.image

import org.jetbrains.kotlinx.dl.datasets.OnHeapDataset
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import java.io.IOException
import java.io.InputStream
import javax.imageio.ImageIO

/** Helper class to convert images to [FloatArray]. */
public class ImageConverter {
    public companion object {
        /** All pixels has values in range [0; 255]. */
        public fun toRawFloatArray(inputStream: InputStream, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return OnHeapDataset.toRawVector(
                toRawPixels(inputStream, colorOrder)
            )
        }

        /** All pixels has values in range [0; 255]. */
        public fun toRawFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return OnHeapDataset.toRawVector(
                toRawPixels(imageFile.inputStream(), colorOrder)
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(
            inputStream: InputStream,
            colorOrder: ColorOrder = ColorOrder.BGR
        ): FloatArray {
            return OnHeapDataset.toNormalizedVector(
                toRawPixels(inputStream, colorOrder)
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return OnHeapDataset.toNormalizedVector(
                toRawPixels(imageFile.inputStream(), colorOrder)
            )
        }

        private fun toRawPixels(inputStream: InputStream, colorOrder: ColorOrder = ColorOrder.BGR): ByteArray {
            val image = getImage(inputStream)

            return imageToByteArray(image, colorOrder)
        }

        private fun imageToByteArray(image: BufferedImage, colorOrder: ColorOrder): ByteArray {
            val res = (image.raster.dataBuffer as DataBufferByte).data // pixels

            if (colorOrder == ColorOrder.BGR) {
                for (i in res.indices) {
                    if (i % 3 == 2) { // swap i and i-2 elements from BGR to RGB
                        val tmp = res[i]
                        res[i] = res[i - 2]
                        res[i - 2] = tmp
                    }
                }
            }
            return res
        }

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

        /**
         * Returns [BufferedImage] extracted from [inputStream] with [imageType].
         */
        @Throws(IOException::class)
        public fun getImage(inputStream: InputStream, imageType: String = "png"): BufferedImage {
            ImageIO.setUseCache(false)
            val image = ImageIO.read(inputStream)
            /*val baos = ByteArrayOutputStream()
            ImageIO.write(image, imageType, baos)*/
            return image
        }
    }
}

/** Represents the order of colors in pixel reading. */
public enum class ColorOrder {
    /** Red, green, blue. */
    RGB,

    /** Blue, green, red. */
    BGR,

    RGBA,

    GREYSCALE
}
