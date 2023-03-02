/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import java.awt.RenderingHints
import java.awt.geom.AffineTransform
import java.awt.image.BufferedImage
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sin

/**
 * This image preprocessor defines the Rotate operation.
 *
 * It rotates the input image on [degrees].
 *
 * The final image quality depends on the following tunable parameters: [interpolation], [renderingSpeed], [enableAntialiasing].
 *
 * @property [degrees] The rotation angle.
 * @property [interpolation] Interpolation algorithm.
 * @property [renderingSpeed] Speed of preprocessing.
 * @property [enableAntialiasing] The image will be cropped from the right by the given number of pixels.
 */
public class Rotate(
    public var degrees: Float = 90f,
    public var interpolation: InterpolationType = InterpolationType.BICUBIC,
    public var renderingSpeed: RenderingSpeed = RenderingSpeed.MEDIUM,
    public var enableAntialiasing: Boolean = true
) : Operation<BufferedImage, BufferedImage> {
    override fun apply(input: BufferedImage): BufferedImage {
        val width: Int = input.width
        val height: Int = input.height
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
            input.type
        )

        val g2d = rotatedImage.createGraphics()

        val renderingHint = when (interpolation) {
            InterpolationType.BILINEAR -> RenderingHints.VALUE_INTERPOLATION_BILINEAR
            InterpolationType.NEAREST -> RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR
            InterpolationType.BICUBIC -> RenderingHints.VALUE_INTERPOLATION_BICUBIC
        }

        val renderingSpeed = when (renderingSpeed) {
            RenderingSpeed.FAST -> RenderingHints.VALUE_RENDER_SPEED
            RenderingSpeed.SLOW -> RenderingHints.VALUE_RENDER_QUALITY
            RenderingSpeed.MEDIUM -> RenderingHints.VALUE_RENDER_DEFAULT
        }

        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION, renderingHint)
        g2d.setRenderingHint(RenderingHints.KEY_RENDERING, renderingSpeed)

        if (enableAntialiasing) {
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON)
        }

        //Rotate the image
        val affineTransform = AffineTransform()
        affineTransform.rotate(theta, centerByX.toDouble(), centerByY.toDouble())
        g2d.transform = affineTransform
        g2d.drawImage(input, -minX, -minY, null)
        g2d.dispose()

        return rotatedImage
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
