/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.image

import org.jetbrains.kotlinx.dl.dataset.OnHeapDataset
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.Convert
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import java.io.IOException
import java.io.InputStream
import javax.imageio.ImageIO


/** Helper object to convert images to [FloatArray]. */
public object ImageConverter {
    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(image: BufferedImage, colorMode: ColorMode? = null): FloatArray {
        return imageToFloatArray(image, colorMode)
    }

    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(inputStream: InputStream, colorMode: ColorMode? = null): FloatArray {
        return toRawFloatArray(toBufferedImage(inputStream), colorMode)
    }

    /** All pixels has values in range [0; 255]. */
    public fun toRawFloatArray(imageFile: File, colorMode: ColorMode? = null): FloatArray {
        return imageFile.inputStream().use { toRawFloatArray(it, colorMode) }
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(image: BufferedImage, colorMode: ColorMode? = null): FloatArray {
        return toRawFloatArray(image, colorMode).also { normalize(it) }
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(inputStream: InputStream, colorMode: ColorMode? = null): FloatArray {
        return toNormalizedFloatArray(toBufferedImage(inputStream), colorMode)
    }

    /** All pixels in range [0;1) */
    public fun toNormalizedFloatArray(imageFile: File, colorMode: ColorMode? = null): FloatArray {
        return imageFile.inputStream().use { toNormalizedFloatArray(it, colorMode) }
    }

    /**
     * Returns [BufferedImage] extracted from [inputStream].
     */
    @Throws(IOException::class)
    public fun toBufferedImage(inputStream: InputStream): BufferedImage {
        ImageIO.setUseCache(false)
        return ImageIO.read(inputStream)
    }

    private val supportedImageTypes = setOf(
        BufferedImage.TYPE_3BYTE_BGR, BufferedImage.TYPE_INT_BGR,
        BufferedImage.TYPE_INT_RGB, BufferedImage.TYPE_BYTE_GRAY
    )

    private fun imageToFloatArray(image: BufferedImage, colorMode: ColorMode?): FloatArray {
        if (colorMode != null && image.colorMode() != colorMode) {
            return imageToFloatArray(Convert(colorMode = colorMode).apply(image))
        }
        return imageToFloatArray(image)
    }

    private fun imageToFloatArray(image: BufferedImage): FloatArray {
        check(image.alphaRaster == null) { "Images with alpha channels are not supported yet!" }
        check(supportedImageTypes.contains(image.type)) {
            "Images with type ${image.type} are not supported yet. " +
                "Supported types are: $supportedImageTypes. See also `java.awt.image.BufferedImage.getType`."
        }

        val raster = image.raster
        if (raster.dataBuffer is DataBufferByte) {
            return OnHeapDataset.toRawVector((raster.dataBuffer as DataBufferByte).data)
        }
        val buffer = FloatArray(raster.numBands * raster.width * raster.height)
        val result = raster.getPixels(0, 0, raster.width, raster.height, buffer)
        if (image.colorMode() == ColorMode.BGR) { // getPixels returns data in RGB order
            swapRandB(result)
        }
        return result
    }

    public fun swapRandB(res: FloatArray) {
        for (i in res.indices step 3) {
            val tmp = res[i]
            res[i] = res[i + 2]
            res[i + 2] = tmp
        }
    }

    private fun normalize(data: FloatArray, scale: Float = 255.0f) {
        for (i in data.indices) data[i] /= scale
    }

    /** Converts [image] with [colorMode] to the 3D array. */
    public fun imageTo3DFloatArray(
        image: BufferedImage,
        colorMode: ColorMode = ColorMode.BGR
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
                if (colorMode == ColorMode.RGB) {
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
                if (colorMode == ColorMode.RGB) {
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

/** Represents the number and order of color channels in the image */
public enum class ColorMode(public val channels: Int) {
    /** Red, green, blue. */
    RGB(3),

    /** Blue, green, red. */
    BGR(3),

    /** Grayscale **/
    GRAYSCALE(1)
}

internal fun BufferedImage.colorMode(): ColorMode {
    return when (type) {
        BufferedImage.TYPE_INT_RGB -> ColorMode.RGB
        BufferedImage.TYPE_3BYTE_BGR, BufferedImage.TYPE_INT_BGR -> ColorMode.BGR
        BufferedImage.TYPE_BYTE_GRAY -> ColorMode.GRAYSCALE
        else -> throw UnsupportedOperationException("Images with type $type are not supported.")
    }
}

public fun ColorMode.imageType(): Int {
    return when (this) {
        ColorMode.RGB -> BufferedImage.TYPE_INT_RGB
        ColorMode.BGR -> BufferedImage.TYPE_3BYTE_BGR
        ColorMode.GRAYSCALE -> BufferedImage.TYPE_BYTE_GRAY
    }
}
