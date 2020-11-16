/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.image

import org.jetbrains.kotlinx.dl.datasets.Dataset
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import javax.imageio.ImageIO

/** Helper class to convert images to [FloatArray]. */
public class ImageConverter {
    public companion object {
        /** All pixels has values in range [0; 255]. */
        public fun toRawFloatArray(inputStream: InputStream, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return Dataset.toRawVector(
                toRawPixels(inputStream, colorOrder)
            )
        }

        /** All pixels has values in range [0; 255]. */
        public fun toRawFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return Dataset.toRawVector(
                toRawPixels(imageFile.inputStream(), colorOrder)
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(
            inputStream: InputStream,
            colorOrder: ColorOrder = ColorOrder.BGR
        ): FloatArray {
            return Dataset.toNormalizedVector(
                toRawPixels(inputStream, colorOrder)
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(imageFile: File, colorOrder: ColorOrder = ColorOrder.BGR): FloatArray {
            return Dataset.toNormalizedVector(
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

        /**
         * Returns [BufferedImage] extracted from [inputStream] with [imageType].
         */
        @Throws(IOException::class)
        public fun getImage(inputStream: InputStream, imageType: String = "png"): BufferedImage {
            val image = ImageIO.read(inputStream)
            val baos = ByteArrayOutputStream()
            ImageIO.write(image, imageType, baos)
            return image
        }
    }
}

/** Represents the order of colors in pixel reading. */
public enum class ColorOrder {
    /** Red, green, blue. */
    RGB,

    /** Blue, green, red. */
    BGR
}
