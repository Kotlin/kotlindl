/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.getShape
import java.awt.Graphics
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
) : ImagePreprocessorBase() {

    override fun getOutputShape(inputShape: ImageShape?): ImageShape? {
        if (inputShape == null) return null
        return ImageShape(
            width = inputShape.width?.minus(left)?.minus(right),
            height = inputShape.height?.minus(top)?.minus(bottom),
            channels = inputShape.channels
        )
    }

    override fun apply(image: BufferedImage): BufferedImage {
        val croppedImageShape = getOutputShape(image.getShape())!!

        val img = image.getSubimage(
            left, top, croppedImageShape.width!!.toInt(), croppedImageShape.height!!.toInt(),
        )

        val croppedImage = BufferedImage(img.width, img.height, image.type)
        val g: Graphics = croppedImage.createGraphics()
        g.drawImage(img, 0, 0, null)

        return croppedImage
    }
}
