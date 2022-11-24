/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter.supportedImageTypes
import org.jetbrains.kotlinx.dl.impl.util.toRawVector
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.File
import java.io.IOException
import java.io.InputStream
import javax.imageio.ImageIO


/**
 * Helper object with methods to convert [BufferedImage] to [FloatArray].
 *
 * @see [supportedImageTypes] for a list of supported image types.
 * */
public object ImageConverter {

    private val supportedImageTypes = setOf(
        BufferedImage.TYPE_3BYTE_BGR, BufferedImage.TYPE_INT_BGR,
        BufferedImage.TYPE_INT_RGB, BufferedImage.TYPE_BYTE_GRAY
    )

    /**
     * Converts [image] to [FloatArray] without normalization.
     *
     * @param [image]       image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in the `[0, 255]` range
     * */
    public fun toRawFloatArray(image: BufferedImage, colorMode: ColorMode? = null): FloatArray {
        return imageToFloatArray(image, colorMode)
    }

    /**
     * Reads the image from [inputStream] and converts it to [FloatArray] without normalization.
     *
     * @param [inputStream] source of the image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in `[0, 255]` range
     * */
    public fun toRawFloatArray(inputStream: InputStream, colorMode: ColorMode? = null): FloatArray {
        return toRawFloatArray(toBufferedImage(inputStream), colorMode)
    }

    /**
     * Reads the image from [imageFile] and converts it to [FloatArray] without normalization.
     *
     * @param [imageFile]   source of the image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in the `[0, 255]` range
     * */
    public fun toRawFloatArray(imageFile: File, colorMode: ColorMode? = null): FloatArray {
        return imageFile.inputStream().use { toRawFloatArray(it, colorMode) }
    }

    /**
     * Converts [image] to [FloatArray] and scales the values, so they would fit into the `[0, 1)` range.
     *
     * @param [image]       image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in the `[0, 1)` range
     * */
    public fun toNormalizedFloatArray(image: BufferedImage, colorMode: ColorMode? = null): FloatArray {
        return toRawFloatArray(image, colorMode).also { normalize(it) }
    }

    /**
     * Reads the image from [inputStream], converts it to [FloatArray] and scales the values,
     * so they would fit into the `[0, 1)` range.
     *
     * @param [inputStream] source of the image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in the `[0, 1)` range
     * */
    public fun toNormalizedFloatArray(inputStream: InputStream, colorMode: ColorMode? = null): FloatArray {
        return toNormalizedFloatArray(toBufferedImage(inputStream), colorMode)
    }

    /**
     * Reads the image from [imageFile], converts it to [FloatArray] and scales the values,
     * so they would fit into the `[0, 1)` range.
     *
     * @param [imageFile]   source of the image to convert
     * @param [colorMode]   color mode to convert the image to. `null` value keeps the original color mode.
     * @return [FloatArray] with pixel values in the `[0, 1)` range
     * */
    public fun toNormalizedFloatArray(imageFile: File, colorMode: ColorMode? = null): FloatArray {
        return imageFile.inputStream().use { toNormalizedFloatArray(it, colorMode) }
    }

    /**
     * Returns [BufferedImage] extracted from [inputStream].
     *
     * @param [inputStream] source of the image
     */
    @Throws(IOException::class)
    public fun toBufferedImage(inputStream: InputStream): BufferedImage {
        ImageIO.setUseCache(false)
        return ImageIO.read(inputStream)
    }

    /**
     * Returns [BufferedImage] extracted from [file].
     *
     * @param [file] source of the image
     */
    @Throws(IOException::class)
    public fun toBufferedImage(file: File): BufferedImage {
        return file.inputStream().use { inputStream -> toBufferedImage(inputStream) }
    }

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
            return toRawVector((raster.dataBuffer as DataBufferByte).data)
        }
        val buffer = FloatArray(raster.numBands * raster.width * raster.height)
        val result = raster.getPixels(0, 0, raster.width, raster.height, buffer)
        if (image.colorMode() == ColorMode.BGR) { // getPixels returns data in RGB order
            swapRandB(result)
        }
        return result
    }

    /**
     * Given a float array representing an image, swaps red and green channels in it.
     *
     * @param [image] image to swap channels in
     */
    public fun swapRandB(image: FloatArray) {
        for (i in image.indices step 3) {
            val tmp = image[i]
            image[i] = image[i + 2]
            image[i + 2] = tmp
        }
    }

    private fun normalize(data: FloatArray, scale: Float = 255.0f) {
        for (i in data.indices) data[i] /= scale
    }

    /**
     * Converts [image] with [colorMode] to the 3D array.
     *
     * @param [image]     image to convert
     * @param [colorMode] color mode used in the target array
     * @return a 3D array with the image in a format `height x width x channels`
     * */
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

    /**
     * Converts [inputArray] of type [FloatArray] to [BufferedImage] with [outputShape] provided.
     *
     * The output [BufferedImage] will have the same ColorMode as [arrayColorMode] of an input tensor.
     *
     * If [isNormalized] is true, then [inputArray] values considered to be in [0..1) interval
     * and will be rescaled to [0..255) interval. Values that are outside this range are clamped
     * to avoid artifacts on the image.
     *
     * If an array requires custom processing, one can use method variation that accepts [ArrayTransform].
     *
     * @param [inputArray]      float array to convert.
     * @param [outputShape]     shape of the output image.
     * @param [arrayColorMode]  [ColorMode] of an [inputArray].
     * @param [isNormalized]    [Boolean] that indicates [inputArray] should be rescaled to [0..255) interval.
     * @return [BufferedImage]  result image.
     * */
    public fun floatArrayToBufferedImage(
        inputArray: FloatArray,
        outputShape: TensorShape,
        arrayColorMode: ColorMode,
        isNormalized: Boolean
    ): BufferedImage {
        return floatArrayToBufferedImage(inputArray, outputShape[0].toInt(), outputShape[1].toInt(), arrayColorMode, isNormalized)
    }

    /**
     * Converts [inputArray] of type [FloatArray] to [BufferedImage] with [width] and [height] provided.
     *
     * The output [BufferedImage] will have the same ColorMode as [arrayColorMode] of an input tensor.
     *
     * If [isNormalized] is true, then [inputArray] values considered to be in [0..1) interval
     * and will be rescaled to [0..255) interval. Values that are outside this range are clamped
     * to avoid artifacts on the image.
     *
     * If an array requires custom processing, one can use method variation that accepts [ArrayTransform].
     *
     * @param [inputArray]      float array to convert.
     * @param [width]           width of the output image.
     * @param [height]          height of the output image.
     * @param [arrayColorMode]  [ColorMode] of an [inputArray].
     * @param [isNormalized]    [Boolean] that indicates [inputArray] should be rescaled to [0..255) interval.
     * @return [BufferedImage]  result image.
     * */
    public fun floatArrayToBufferedImage(
        inputArray: FloatArray,
        width: Int,
        height: Int,
        arrayColorMode: ColorMode,
        isNormalized: Boolean
    ): BufferedImage {
        return floatArrayToBufferedImage(inputArray, width, height, arrayColorMode) {
            if (isNormalized) denormalizeInplace(it, scale = 255f)
            for ((i, value) in it.withIndex()) {
                if (value < 0) it[i] = 0f
                if (value > 255) it[i] = 255f
            }
            it
        }
    }

    /**
     * Converts [inputArray] of type [FloatArray] to [BufferedImage] with [width] and [height] provided.
     * If a custom [arrayTransform] is needed, lambda or ArrayTransform can be provided.
     *
     * The output [BufferedImage] will have the same ColorMode as [arrayColorMode] of an input tensor.
     *
     * If an [arrayTransform] is given, then [arrayColorMode] is interpreted as a color mode of an array after transform.
     *
     * [inputArray] is explicitly copied before applying [arrayTransform].
     *
     * @see ArrayTransform
     *
     * @param [inputArray]      float array to convert.
     * @param [width]           shape of the output image.
     * @param [height]          height of the output image.
     * @param [arrayColorMode]  [ColorMode] of an [inputArray].
     * @param [arrayTransform]  [ArrayTransform] implementation.
     *                          Thanks to Kotlin SAM convention it can be supplied as [(FloatArray) -> FloatArray] lambda.
     * @return [BufferedImage]  result image.
     * */
    public fun floatArrayToBufferedImage(
        inputArray: FloatArray,
        width: Int,
        height: Int,
        arrayColorMode: ColorMode,
        arrayTransform: ArrayTransform? = null
    ): BufferedImage {
        val dataCopy = arrayTransform?.invoke(inputArray.copyOf()) ?: inputArray.copyOf()

        require(width * height * arrayColorMode.channels == dataCopy.size) {
            "Requested image shape [$width x $height] does not match with input array size [${inputArray.size}]"
        }

        /* This swap is needed because BufferedImage.raster.setPixels accepts data in RGB format */
        if (arrayColorMode == ColorMode.BGR) {
            swapRandB(dataCopy)
        }

        val output = BufferedImage(width, height, arrayColorMode.imageType())
        output.raster.setPixels(0, 0, output.width, output.height, dataCopy)

        return output
    }
}

internal fun BufferedImage.colorMode(): ColorMode {
    return when (type) {
        BufferedImage.TYPE_INT_RGB -> ColorMode.RGB
        BufferedImage.TYPE_3BYTE_BGR, BufferedImage.TYPE_INT_BGR -> ColorMode.BGR
        BufferedImage.TYPE_BYTE_GRAY -> ColorMode.GRAYSCALE
        else -> throw UnsupportedOperationException("Images with type $type are not supported.")
    }
}

/**
 * Returns an integer representing a type of [BufferedImage] corresponding to this color mode.
 */
public fun ColorMode.imageType(): Int {
    return when (this) {
        ColorMode.RGB -> BufferedImage.TYPE_INT_RGB
        ColorMode.BGR -> BufferedImage.TYPE_3BYTE_BGR
        ColorMode.GRAYSCALE -> BufferedImage.TYPE_BYTE_GRAY
    }
}
