/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.image.copy
import java.awt.image.BufferedImage

/**
 * This image preprocessor defines centerCrop operation.
 * It crops the given image at the center. If the image size is smaller than the output size along any edge,
 * the image is padded with 0 and then center cropped.
 *
 * @property [size] target image size.
 */
public class CenterCrop(public var size: Int = -1) : ImageOperationBase() {
    override fun apply(input: BufferedImage): BufferedImage {
        if (size <= 0 || (input.width == size && input.height == size)) return input

        val paddedImage = padIfNecessary(input)
        val result = paddedImage.getSubimage(
            (paddedImage.width - size) / 2,
            (paddedImage.height - size) / 2,
            size, size
        ).copy()

        save?.save("center_crop_result", result)
        return result
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        if (size <= 0) return inputShape
        return TensorShape(size.toLong(), size.toLong(), inputShape[2])
    }
    private fun padIfNecessary(image: BufferedImage): BufferedImage {
        if (image.width < size || image.height < size) {
            val verticalSpace = (size - image.height).coerceAtLeast(0)
            val horizontalSpace = (size - image.width).coerceAtLeast(0)
            val top = verticalSpace / 2
            val left = horizontalSpace / 2
            return Padding(
                top = top, bottom = verticalSpace - top,
                left = left, right = horizontalSpace - left,
                mode = PaddingMode.Black
            ).apply(image)
        }
        return image
    }
}
