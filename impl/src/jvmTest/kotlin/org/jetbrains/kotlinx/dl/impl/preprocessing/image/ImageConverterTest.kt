/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.impl.preprocessing.PreprocessingImageTest.Companion.setRGB
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
import kotlin.math.round

class ImageConverterTest {
    @Test
    fun bgr2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_3BYTE_BGR, ColorMode.BGR)

    @Test
    fun bgr2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_3BYTE_BGR, ColorMode.RGB)

    @Test
    fun ibgr2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_BGR, ColorMode.BGR)

    @Test
    fun ibgr2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_BGR, ColorMode.RGB)

    @Test
    fun rgb2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_RGB, ColorMode.BGR)

    @Test
    fun rgb2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_RGB, ColorMode.RGB)

    private fun imageToFloatsTest(sourceImageType: Int, targetColorMode: ColorMode) {
        val sourceImage = BufferedImage(2, 2, sourceImageType)
        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)
        val color4 = Color(210, 160, 60)
        sourceImage.setRGB(0, 0, color1.rgb)
        sourceImage.setRGB(0, 1, color2.rgb)
        sourceImage.setRGB(1, 0, color3.rgb)
        sourceImage.setRGB(1, 1, color4.rgb)

        val targetImage = ImageConverter.toNormalizedFloatArray(sourceImage, targetColorMode)

        val expectedImage = FloatArray(sourceImage.getShape().numElements().toInt()) { 0f }
        expectedImage.setRGB(0, 0, color1, sourceImage.getShape(), targetColorMode)
        expectedImage.setRGB(0, 1, color2, sourceImage.getShape(), targetColorMode)
        expectedImage.setRGB(1, 0, color3, sourceImage.getShape(), targetColorMode)
        expectedImage.setRGB(1, 1, color4, sourceImage.getShape(), targetColorMode)
        Assertions.assertArrayEquals(expectedImage, targetImage)
    }

    @Test
    fun grayscaleImageToFloatsTest() {
        val expectedImage = floatArrayOf(0.2f, 0.4f, 0.6f, 0.8f)

        val sourceImage = BufferedImage(2, 2, BufferedImage.TYPE_BYTE_GRAY)
        sourceImage.raster.setDataElements(0, 0, gray(expectedImage[0]))
        sourceImage.raster.setDataElements(1, 0, gray(expectedImage[1]))
        sourceImage.raster.setDataElements(0, 1, gray(expectedImage[2]))
        sourceImage.raster.setDataElements(1, 1, gray(expectedImage[3]))

        val targetImage = ImageConverter.toNormalizedFloatArray(sourceImage)
        Assertions.assertArrayEquals(expectedImage, targetImage)
    }

    @Test
    fun floatArrayToBufferedImageCustomConversionTest() {
        val sourceImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)
        val color4 = Color(210, 160, 60)
        sourceImage.setRGB(0, 0, color1.rgb)
        sourceImage.setRGB(0, 1, color2.rgb)
        sourceImage.setRGB(1, 0, color3.rgb)
        sourceImage.setRGB(1, 1, color4.rgb)

        val sourceArray = ImageConverter.toRawFloatArray(sourceImage)
        val tfNormalized = sourceArray.map { v -> v / 127.5f - 1 }.toFloatArray()

        val targetImage = ImageConverter.floatArrayToBufferedImage(
            tfNormalized,
            2, 2,
            arrayColorMode = ColorMode.BGR
        ) {
            it.forEachIndexed { idx, v -> it[idx] = round((v + 1) * 127.5f) }
            it
        }

        Assertions.assertArrayEquals(
            sourceArray,
            ImageConverter.toRawFloatArray(targetImage),
            "Custom of array TF normalized array to image failed"
        )
    }

    private val imageTypes =
        listOf(
            BufferedImage.TYPE_BYTE_GRAY,
            BufferedImage.TYPE_3BYTE_BGR,
            BufferedImage.TYPE_INT_BGR,
            BufferedImage.TYPE_INT_RGB
        )

    @Test
    fun normalizedFloatArrayToBufferedImageTest() {
        for (sourceImageType in imageTypes) {
            val sourceImage = BufferedImage(2, 2, sourceImageType)
            val color1 = Color(50, 150, 200)
            val color2 = Color(10, 190, 70)
            val color3 = Color(210, 40, 40)
            val color4 = Color(210, 160, 60)
            sourceImage.setRGB(0, 0, color1.rgb)
            sourceImage.setRGB(0, 1, color2.rgb)
            sourceImage.setRGB(1, 0, color3.rgb)
            sourceImage.setRGB(1, 1, color4.rgb)

            val sourceArray = ImageConverter.toNormalizedFloatArray(sourceImage)

            val targetImage = ImageConverter.floatArrayToBufferedImage(
                sourceArray,
                2, 2,
                sourceImage.colorMode(),
                isNormalized = true
            )

            Assertions.assertArrayEquals(
                sourceArray,
                ImageConverter.toNormalizedFloatArray(targetImage),
                "Conversion of array normalized array to image in ${sourceImage.colorMode()} mode failed"
            )
        }
    }

    @Test
    fun floatArrayToBufferedImageTest() {
        for (sourceImageType in imageTypes) {
            val sourceImage = BufferedImage(2, 2, sourceImageType)
            val color1 = Color(50, 150, 200)
            val color2 = Color(10, 190, 70)
            val color3 = Color(210, 40, 40)
            val color4 = Color(210, 160, 60)
            sourceImage.setRGB(0, 0, color1.rgb)
            sourceImage.setRGB(0, 1, color2.rgb)
            sourceImage.setRGB(1, 0, color3.rgb)
            sourceImage.setRGB(1, 1, color4.rgb)

            val sourceArray = ImageConverter.toRawFloatArray(sourceImage)

            val targetImage = ImageConverter.floatArrayToBufferedImage(
                sourceArray,
                2, 2,
                sourceImage.colorMode(),
                isNormalized = false
            )

            Assertions.assertArrayEquals(
                sourceArray,
                ImageConverter.toRawFloatArray(targetImage),
                "Conversion of array normalized array to image in ${sourceImage.colorMode()} mode failed"
            )
        }
    }

    @Test
    fun floatArrayToBufferedImageClampTest() {
        val colorMode = ColorMode.RGB
        val expectedImage = BufferedImage(2, 2, colorMode.imageType())
        expectedImage.setRGB(0, 0, Color.black.rgb)
        expectedImage.setRGB(0, 1, Color.white.rgb)
        expectedImage.setRGB(1, 0, Color.black.rgb)
        expectedImage.setRGB(1, 1, Color.white.rgb)

        val inputArray = FloatArray(expectedImage.width * expectedImage.height * colorMode.channels)
        inputArray.fill(-5f, 0, inputArray.size / 2)
        inputArray.fill(300f, inputArray.size / 2, inputArray.size)

        val outputImage = ImageConverter.floatArrayToBufferedImage(
            inputArray, expectedImage.getShape(), colorMode, false
        )
        Assertions.assertArrayEquals(
            ImageConverter.toRawFloatArray(expectedImage),
            ImageConverter.toRawFloatArray(outputImage),
        )
    }

    private fun gray(value: Float) = byteArrayOf((value * 255).toInt().toByte())
}