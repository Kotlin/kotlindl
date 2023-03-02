/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import java.awt.image.BufferedImage

/**
 * Converts provided image to the desired [ColorMode].
 *
 * @property [colorMode] target color mode.
 */
public class Convert(public var colorMode: ColorMode = ColorMode.BGR) : Operation<BufferedImage, BufferedImage> {
    override fun apply(input: BufferedImage): BufferedImage {
        if (input.colorMode() == colorMode) return input
        val outputType = colorMode.imageType()
        val result = BufferedImage(input.width, input.height, outputType)
        val graphics = result.createGraphics()
        graphics.drawImage(input, 0, 0, null)
        graphics.dispose()

        return result
    }

    /**
     * Takes result color mode into account when computing output shape.
     */
    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return when (inputShape.rank()) {
            2, 3 -> TensorShape(inputShape[0], inputShape[1], colorMode.channels.toLong())
            else -> throw IllegalArgumentException("Input shape must have 2 or 3 dimensions")
        }
    }
}
