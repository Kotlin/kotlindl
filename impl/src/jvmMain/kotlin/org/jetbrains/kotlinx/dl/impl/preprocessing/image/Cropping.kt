/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import java.awt.image.BufferedImage

/**
 * This image preprocessor defines the Crop operation.
 *
 * Crop operations creates new image with the following sizes:
 *  - croppedWidth = initialWidth - [left] - [right]
 *  - croppedHeight = initialHeight - [top] - [bottom]
 *
 * @property [top] The image will be cropped from the top by the given number of pixels.
 * @property [bottom] The image will be cropped from the bottom by the given number of pixels.
 * @property [left] The image will be cropped from the left by the given number of pixels.
 * @property [right] The image will be cropped from the right by the given number of pixels.
 */
public class Cropping(
    public var top: Int = 1,
    public var bottom: Int = 1,
    public var left: Int = 1,
    public var right: Int = 1
) : Operation<BufferedImage, BufferedImage> {
    override fun apply(input: BufferedImage): BufferedImage {
        val croppedImageShape = getOutputShape(input.getShape())
        val (width, height, _) = croppedImageShape.dims()

        return input.getSubimage(
            left, top,
            width.toInt(),
            height.toInt()
        ).copy()
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        val outputWidth = if (inputShape[0] == -1L) -1 else inputShape[0] - left - right
        val outputHeight = if (inputShape[1] == -1L) -1 else inputShape[1] - top - bottom

        return when (inputShape.rank()) {
            2 -> TensorShape(outputWidth, outputHeight)
            3 -> TensorShape(outputWidth, outputHeight, inputShape[2])
            else -> throw IllegalArgumentException("Cropping operation is applicable only to images with rank 2 or 3")
        }
    }
}
