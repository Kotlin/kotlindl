/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage

class PreprocessingTensorTest {
    @Test
    fun normalizeMeanAndStdTest() {
        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color(50, 150, 200).rgb)
        inputImage.setRGB(0, 1, Color(10, 190, 70).rgb)
        inputImage.setRGB(1, 0, Color(210, 40, 40).rgb)
        inputImage.setRGB(1, 1, Color(210, 160, 60).rgb)

        val imageFloats = ImageConverter.toNormalizedFloatArray(inputImage)

        val (normalizedImage, _) = Normalizing().apply {
            mean = imageFloats.mean(channels = 3)
            std = imageFloats.std(channels = 3)
            channelsLast = true
        }.apply(imageFloats to TensorShape(2, 2, 3))

        Assertions.assertArrayEquals(FloatArray(3) { 0f }, normalizedImage.mean(3), EPS)
        Assertions.assertArrayEquals(FloatArray(3) { 1f }, normalizedImage.std(3), EPS)
    }

    @Test
    fun normalizeTest() {
        val input = floatArrayOf(20f, 30f, 40f, 50f, 60f, 70f)
        val meanValues = floatArrayOf(10f, 20f, 30f)
        val stdValues = floatArrayOf(10f, 10f, 10f)

        val (channelsFirst, _) = Normalizing().apply {
            mean = meanValues
            std = stdValues
            channelsLast = false
        }.apply(input.copyOf() to TensorShape(3, 1, 2))

        val (channelsLast, _) = Normalizing().apply {
            mean = meanValues
            std = stdValues
            channelsLast = true
        }.apply(input.copyOf() to TensorShape(1, 2, 3))

        Assertions.assertArrayEquals(floatArrayOf(1f, 2f, 2f, 3f, 3f, 4f), channelsFirst, EPS)
        Assertions.assertArrayEquals(floatArrayOf(1f, 1f, 1f, 4f, 4f, 4f), channelsLast, EPS)
    }

    companion object {
        private const val EPS: Float = 2e-7f
    }
}