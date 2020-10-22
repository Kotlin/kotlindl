/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package datasets.image

import datasets.Dataset
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
        public fun toRawFloatArray(inputStream: InputStream, imageType: ImageType = ImageType.JPG): FloatArray {
            return Dataset.toRawVector(
                toRawPixels(inputStream)
            )
        }

        /** All pixels has values in range [0; 255]. */
        public fun toRawFloatArray(imageFile: File, imageType: ImageType = ImageType.JPG): FloatArray {
            return Dataset.toRawVector(
                toRawPixels(imageFile.inputStream())
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(inputStream: InputStream, imageType: ImageType = ImageType.JPG): FloatArray {
            return Dataset.toNormalizedVector(
                toRawPixels(inputStream)
            )
        }

        /** All pixels in range [0;1) */
        public fun toNormalizedFloatArray(imageFile: File, imageType: ImageType = ImageType.JPG): FloatArray {
            return Dataset.toNormalizedVector(
                toRawPixels(imageFile.inputStream())
            )
        }

        private fun toRawPixels(inputStream: InputStream): ByteArray {
            val image = getImage(inputStream)

            return (image.raster.dataBuffer as DataBufferByte).data // pixels
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

/** */
public enum class ImageType {
    /** */
    JPG,

    /** */
    PNG,

    /** */
    GIF
}