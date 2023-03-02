/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.*
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Test
import java.awt.image.BufferedImage

class PreprocessingFinalShapeTest {
    @Test
    fun resizeNoInputShape() {
        val preprocess = pipeline<BufferedImage>()
            .resize {
                outputWidth = 100
                outputHeight = 100
                interpolation = InterpolationType.NEAREST
            }
            .toFloatArray { }
        assertEquals(TensorShape(100, 100, -1), preprocess.getOutputShape(TensorShape(-1, -1, -1)))
    }

    @Test
    fun resizeInputShape() {
        val preprocess = pipeline<BufferedImage>()
            .resize {
                outputWidth = 100
                outputHeight = 100
                interpolation = InterpolationType.NEAREST
            }
            .toFloatArray { }
        assertEquals(TensorShape(100, 100, 3), preprocess.getOutputShape(TensorShape(20, 20, 3)))
    }

    @Test
    fun cropImage() {
        val preprocess = pipeline<BufferedImage>()
            .crop {
                left = 3
                right = 11
                top = 5
                bottom = 7
            }
            .toFloatArray { }
        assertEquals(TensorShape(186, 188, 3), preprocess.getOutputShape(TensorShape(200, 200, 3)))
    }

    @Test
    fun cropTwice() {
        val preprocess = pipeline<BufferedImage>()
            .crop {
                left = 3
                right = 11
                top = 5
                bottom = 7
            }
            .crop {
                left = 4
                right = 2
                top = 5
                bottom = 3
            }
            .toFloatArray { }
        assertEquals(TensorShape(180, 180, 3), preprocess.getOutputShape(TensorShape(200, 200, 3)))
    }

    @Test
    fun resizeAndCrop() {
        val preprocess = pipeline<BufferedImage>()
            .resize {
                outputWidth = 150
                outputHeight = 100
                interpolation = InterpolationType.NEAREST
            }
            .crop {
                left = 5
                right = 5
                top = 5
                bottom = 5
            }
            .toFloatArray { }
        assertEquals(TensorShape(140, 90, 3), preprocess.getOutputShape(TensorShape(200, 200, 3)))
    }

    @Test
    fun rotateImage() {
        val preprocess = pipeline<BufferedImage>()
            .rotate {
                degrees = 30f
            }
            .toFloatArray { }
        assertEquals(TensorShape(200, 200, 3), preprocess.getOutputShape(TensorShape(200, 200, 3)))
    }

    @Test
    fun padImage() {
        val preprocess = pipeline<BufferedImage>()
            .pad {
                top = 5
                bottom = 7
                left = 11
                right = 13
            }
            .toFloatArray { }
        assertEquals(TensorShape(324, 212, 1), preprocess.getOutputShape(TensorShape(300, 200, 1)))
    }

    @Test
    fun centerCropImage() {
        val preprocess = pipeline<BufferedImage>()
            .centerCrop { size = 15 }
            .toFloatArray { }
        assertEquals(TensorShape(15, 15, 1), preprocess.getOutputShape(TensorShape(10, 20, 1)))
    }

    @Test
    fun convertImageToGrayscaleTest() {
        val preprocess = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.GRAYSCALE }
            .toFloatArray { }

        val image = BufferedImage(10, 20, BufferedImage.TYPE_3BYTE_BGR)
        val (_, actualShape) = preprocess.apply(image)

        assertEquals(actualShape, preprocess.getOutputShape(TensorShape(10, 20, 1)))
    }

    @Test
    fun convertImageToRGBTest() {
        val preprocess = pipeline<BufferedImage>()
            .convert { colorMode = ColorMode.RGB }
            .toFloatArray { }

        val image = BufferedImage(10, 20, BufferedImage.TYPE_3BYTE_BGR)
        val (_, actualShape) = preprocess.apply(image)

        assertEquals(actualShape, preprocess.getOutputShape(TensorShape(10, 20, 3)))
    }
}
