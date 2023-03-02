/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.jetbrains.kotlinx.dl.impl.util.set3D
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage
import kotlin.math.roundToInt

class PreprocessingImageTest {
    @Test
    fun resizeTest() {
        val preprocess = pipeline<BufferedImage>()
            .resize {
                outputWidth = 4
                outputHeight = 4
                interpolation = InterpolationType.NEAREST
            }
            .toFloatArray { }
            .rescale { }

        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color.BLUE.rgb)
        inputImage.setRGB(1, 1, Color.RED.rgb)
        val (imageFloats, tensorShape) = preprocess.apply(inputImage)
        Assertions.assertEquals(TensorShape(4, 4, 3), tensorShape)
        val expectedImage = FloatArray(tensorShape.numElements().toInt()) { 0f }.apply {
            for (i in 0..1)
                for (j in 0..1)
                    setRGB(i, j, Color.BLUE, tensorShape, ColorMode.BGR)
            for (i in 2..3)
                for (j in 2..3)
                    setRGB(i, j, Color.RED, tensorShape, ColorMode.BGR)
        }
        Assertions.assertArrayEquals(expectedImage, imageFloats)
    }

    @Test
    fun cropTest() {
        val preprocess = pipeline<BufferedImage>()
            .crop {
                left = 1
                right = 0
                top = 0
                bottom = 1
            }
            .toFloatArray { }
            .rescale { }

        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color.BLUE.rgb)
        inputImage.setRGB(1, 0, Color.GREEN.rgb)
        inputImage.setRGB(1, 1, Color.RED.rgb)
        inputImage.setRGB(0, 1, Color.GREEN.rgb)

        val (imageFloats, tensorShape) = preprocess.apply(inputImage)
        Assertions.assertEquals(TensorShape(1, 1, 3), tensorShape)

        val expectedImage = FloatArray(3).apply { setRGB(0, 0, Color.GREEN, tensorShape, ColorMode.BGR) }
        Assertions.assertArrayEquals(expectedImage, imageFloats)
    }

    @Test
    fun rotateTest() {
        val preprocess = pipeline<BufferedImage>()
            .rotate {
                degrees = 90f
            }
            .toFloatArray { }
            .rescale { }
        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color.BLUE.rgb)
        inputImage.setRGB(1, 1, Color.RED.rgb)
        val (imageFloats, tensorShape) = preprocess.apply(inputImage)
        Assertions.assertEquals(TensorShape(2, 2, 3), tensorShape)
        val expectedImage = FloatArray(tensorShape.numElements().toInt()) { 0f }
        expectedImage.setRGB(1, 0, Color.BLUE, tensorShape, ColorMode.BGR)
        expectedImage.setRGB(0, 1, Color.RED, tensorShape, ColorMode.BGR)
        Assertions.assertArrayEquals(expectedImage, imageFloats)
    }

    @Test
    fun constantPaddingTest() {
        val preprocess = pipeline<BufferedImage>()
            .pad {
                top = 1
                bottom = 2
                left = 3
                right = 4
                mode = PaddingMode.Fill(Color.GRAY)
            }
            .toFloatArray { }
            .rescale { }

        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color.BLUE.rgb)
        inputImage.setRGB(1, 1, Color.RED.rgb)
        val (imageFloats, tensorShape) = preprocess.apply(inputImage)

        Assertions.assertEquals(TensorShape(9, 5, 3), tensorShape)

        val expectedImage = FloatArray(tensorShape.numElements().toInt()) { Color.GRAY.red / 255f }
        expectedImage.setRGB(3, 1, Color.BLUE, tensorShape, ColorMode.BGR)
        expectedImage.setRGB(4, 1, Color.BLACK, tensorShape, ColorMode.BGR)
        expectedImage.setRGB(4, 2, Color.RED, tensorShape, ColorMode.BGR)
        expectedImage.setRGB(3, 2, Color.BLACK, tensorShape, ColorMode.BGR)

        Assertions.assertArrayEquals(expectedImage, imageFloats)

    }

    @Test
    fun convertTest() {
        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color.BLUE.rgb)
        inputImage.setRGB(1, 1, Color.RED.rgb)
        val rgbImage = Convert(colorMode = ColorMode.RGB).apply(inputImage)
        val rgbImageFloats = ImageConverter.toNormalizedFloatArray(rgbImage)

        val tensorShape = TensorShape(2, 2, 3)
        val expectedImageFloats = FloatArray(tensorShape.numElements().toInt()) { 0f }
        expectedImageFloats.setRGB(0, 0, Color.BLUE, tensorShape, ColorMode.RGB)
        expectedImageFloats.setRGB(1, 1, Color.RED, tensorShape, ColorMode.RGB)
        Assertions.assertArrayEquals(expectedImageFloats, rgbImageFloats)
    }

    @Test
    fun grayscaleTest() {
        val preprocess = pipeline<BufferedImage>()
            .grayscale()
            .toFloatArray { }
            .rescale { }

        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)
        val color4 = Color(210, 160, 60)
        inputImage.setRGB(0, 0, color1.rgb)
        inputImage.setRGB(0, 1, color2.rgb)
        inputImage.setRGB(1, 0, color3.rgb)
        inputImage.setRGB(1, 1, color4.rgb)
        val (imageFloats, tensorShape) = preprocess.apply(inputImage)

        Assertions.assertEquals(TensorShape(2, 2, 1), tensorShape)
        val expectedImage = FloatArray(tensorShape.numElements().toInt()) { 0f }
        expectedImage.setRGB(0, 0, color1, tensorShape, ColorMode.GRAYSCALE)
        expectedImage.setRGB(0, 1, color2, tensorShape, ColorMode.GRAYSCALE)
        expectedImage.setRGB(1, 0, color3, tensorShape, ColorMode.GRAYSCALE)
        expectedImage.setRGB(1, 1, color4, tensorShape, ColorMode.GRAYSCALE)
        Assertions.assertArrayEquals(expectedImage, imageFloats)
    }

    @Test
    fun centerCropTest() {
        val preprocess = pipeline<BufferedImage>()
            .centerCrop { size = 2 }
            .toFloatArray { }
            .rescale { }

        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)

        val inputImage = BufferedImage(1, 3, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, color1.rgb)
        inputImage.setRGB(0, 1, color2.rgb)
        inputImage.setRGB(0, 2, color3.rgb)

        val (imageFloats, tensorShape) = preprocess.apply(inputImage)
        Assertions.assertEquals(TensorShape(2, 2, 3), tensorShape)

        val expectedImage = FloatArray(12)
        expectedImage.setRGB(0, 0, color1, tensorShape, ColorMode.BGR)
        expectedImage.setRGB(0, 1, color2, tensorShape, ColorMode.BGR)
        Assertions.assertArrayEquals(expectedImage, imageFloats)
    }

    companion object {
        internal fun FloatArray.setRGB(x: Int, y: Int, color: Color, tensorShape: TensorShape, colorMode: ColorMode) {
            val colorComponents = when (colorMode) {
                ColorMode.RGB -> floatArrayOf(color.red / 255f, color.green / 255f, color.blue / 255f)
                ColorMode.BGR -> floatArrayOf(color.blue / 255f, color.green / 255f, color.red / 255f)
                ColorMode.GRAYSCALE -> {
                    floatArrayOf(
                        ((0.299 * color.red) +
                                (0.587 * color.green) +
                                (0.114 * color.blue)).roundToInt() / 255f
                    )
                }
            }
            for (i in colorComponents.indices) {
                set3D(y, x, i, tensorShape[0].toInt(), colorMode.channels, colorComponents[i])
            }
        }
    }
}
