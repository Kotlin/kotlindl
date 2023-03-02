/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

/**
 * Decodes an ARGB8888 encoded pixel array to a float array containing the red, green, blue components in NCWH layout.
 */
public fun argB8888ToNCHWArray(encodedPixels: IntArray, width: Int, height: Int, channels: Int): FloatArray {
    val output = FloatArray(width * height * channels)
    val stride = width * height

    for (i in 0 until width) {
        for (j in 0 until height) {
            val idx = height * i + j
            val pixelValue = encodedPixels[idx]

            val (r, g, b) = decodeARGB8888Pixel(pixelValue)

            output[idx] = r
            output[idx + stride] = g
            output[idx + stride * 2] = b
        }
    }

    return output
}

/**
 * Decodes an ARGB8888 encoded pixel array to a float array containing the red, green, blue components in NHWC layout.
 */
public fun argB8888ToNHWCArray(encodedPixels: IntArray, width: Int, height: Int, channels: Int): FloatArray {
    val output = FloatArray(width * height * channels)

    var position = 0
    for (pixelValue in encodedPixels) {
        val (r, g, b) = decodeARGB8888Pixel(pixelValue)

        output[position++] = r
        output[position++] = g
        output[position++] = b
    }

    return output
}

/**
 * Decodes an ARGB8888 encoded pixel to a red, green, blue components.
 */
public fun decodeARGB8888Pixel(pixelValue: Int): Triple<Float, Float, Float> {
    val r = (pixelValue shr 16 and 0xFF).toFloat()
    val g = (pixelValue shr 8 and 0xFF).toFloat()
    val b = (pixelValue and 0xFF).toFloat()

    return Triple(r, g, b)
}
