/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage

class ImageFormatDecodingTestSuite {
    private var encodedPixels: IntArray
    private val w = 2
    private val h = 2

    init {
        val sourceImage = BufferedImage(w, h, BufferedImage.TYPE_INT_ARGB)
        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)
        val color4 = Color(210, 160, 60)
        sourceImage.setRGB(0, 0, color1.rgb)
        sourceImage.setRGB(0, 1, color2.rgb)
        sourceImage.setRGB(1, 0, color3.rgb)
        sourceImage.setRGB(1, 1, color4.rgb)

        encodedPixels = IntArray(w * h)
        for (x in 0 until w) {
            for (y in 0 until h) {
                encodedPixels[y * w + x] = sourceImage.getRGB(x, y)
            }
        }
    }

    @Test
    fun argb8888ToNCHWTest() {
        val nchw = argB8888ToNCHWArray(encodedPixels, w, h, 3)

        assert(
            nchw.contentEquals(
                floatArrayOf(50.0f, 210.0f, 10.0f, 210.0f, 150.0f, 40.0f, 190.0f, 160.0f, 200.0f, 40.0f, 70.0f, 60.0f)
            )
        )
    }

    @Test
    fun argb8888ToNHWCTest() {
        val nchw = argB8888ToNHWCArray(encodedPixels, w, h, 3)

        assert(
            nchw.contentEquals(
                floatArrayOf(50.0f, 150.0f, 200.0f, 210.0f, 40.0f, 40.0f, 10.0f, 190.0f, 70.0f, 210.0f, 160.0f, 60.0f)
            )
        )
    }
}
