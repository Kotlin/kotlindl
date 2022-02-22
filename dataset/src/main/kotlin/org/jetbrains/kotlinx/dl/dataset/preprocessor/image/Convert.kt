/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.colorMode
import org.jetbrains.kotlinx.dl.dataset.image.imageType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.image.BufferedImage

/**
 * Converts provided image to the desired [ColorMode].
 *
 * @property [colorMode] target color mode.
 */
public class Convert(public var colorMode: ColorMode = ColorMode.BGR) : ImagePreprocessorBase() {
    override fun getOutputShape(inputShape: ImageShape): ImageShape {
        return ImageShape(inputShape.width, inputShape.height, colorMode.channels.toLong())
    }

    override fun apply(image: BufferedImage): BufferedImage {
        if (image.colorMode() == colorMode) return image
        val outputType = colorMode.imageType()
        val result = BufferedImage(image.width, image.height, outputType)
        val graphics = result.createGraphics()
        graphics.drawImage(image, 0, 0, null)
        graphics.dispose()
        return result
    }
}