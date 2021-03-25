/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

public class Rotate(public var degrees: Float = 90f) : ImagePreprocessor {
    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        val width: Int = inputShape.width.toInt()
        val height: Int = inputShape.height.toInt()
        var centerByX = width / 2
        var centerByY = height / 2

        val rect = intArrayOf(0, 0, width, 0, width, height, 0, height)

        var minX: Int
        var minY: Int
        var maxX: Int
        var maxY: Int
        minX = centerByX.also { maxX = it }
        minY = centerByY.also { maxY = it }

        val theta = Math.toRadians(degrees.toDouble())


        var i = 0
        while (i < rect.size) {
            val x = (cos(theta) * (rect[i] - centerByX) -
                    sin(theta) * (rect[i + 1] - centerByY) + centerByX).roundToInt()
            val y =
                (sin(theta) * (rect[i] - centerByX) + cos(theta) * (rect[i + 1] - centerByY) + centerByY).roundToInt()

            if (x > maxX) maxX = x
            if (x < minX) minX = x
            if (y > maxY) maxY = y
            if (y < minY) minY = y
            i += 2
        }

        centerByX = (centerByX - minX)
        centerByY = (centerByY - minY)

        val rotatedImage = BufferedImage(
            maxX - minX, maxY - minY,
            BufferedImage.TYPE_3BYTE_BGR
        )

        val g2d = rotatedImage.createGraphics()

        g2d.setRenderingHint(
            RenderingHints.KEY_ANTIALIASING,
            RenderingHints.VALUE_ANTIALIAS_ON
        )

        g2d.setRenderingHint(
            RenderingHints.KEY_INTERPOLATION,
            RenderingHints.VALUE_INTERPOLATION_BICUBIC
        )

        //Rotate the image
        val affineTransform = AffineTransform()
        affineTransform.rotate(theta, centerByX.toDouble(), centerByY.toDouble())
        g2d.transform = affineTransform
        g2d.drawImage(image, -minX, -minY, null)
        g2d.dispose()

        return Pair(rotatedImage, inputShape)
    }
}
