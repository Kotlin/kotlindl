/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import java.awt.Color
import java.awt.image.BufferedImage

/**
 * This image preprocessor defines the Pad operation.
 *
 * Pad operation pads the given image on all sides according to the provided [PaddingMode].
 *
 * @property [top]    The number of pixels to add to the top.
 * @property [bottom] The number of pixels to add to the bottom.
 * @property [left]   The number of pixels to add to the left.
 * @property [right]  The number of pixels to add to the right.
 * @property [mode]   The kind of padding to use.
 */
public class Padding(
    public var top: Int = 0,
    public var bottom: Int = 0,
    public var left: Int = 0,
    public var right: Int = 0,
    public var mode: PaddingMode = PaddingMode.Black
) : Operation<BufferedImage, BufferedImage> {
    override fun apply(input: BufferedImage): BufferedImage {
        val result = BufferedImage(input.width + left + right, input.height + top + bottom, input.type)
        result.draw { graphics2D ->
            graphics2D.drawImage(input, left, top, null)
            when (mode) {
                is PaddingMode.Fill -> {
                    graphics2D.color = (mode as PaddingMode.Fill).color
                    graphics2D.fillRect(0, 0, result.width, top)
                    graphics2D.fillRect(0, result.height - bottom, result.width, bottom)
                    graphics2D.fillRect(0, top, left, result.height - top - bottom)
                    graphics2D.fillRect(result.width - right, top, right, result.height - top - bottom)
                }
            }
        }

        return result
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        val outputWidth = if (inputShape[0] == -1L) -1 else inputShape[0] + left + right
        val outputHeight = if (inputShape[1] == -1L) -1 else inputShape[1] + top + bottom

        return when (inputShape.rank()) {
            2 -> TensorShape(outputWidth, outputHeight)
            3 -> TensorShape(outputWidth, outputHeight, inputShape[2])
            else -> throw IllegalArgumentException("Padding operation is supported only for 2D and 3D tensors")
        }
    }
}

/**
 * Type of padding to use.
 */
public sealed interface PaddingMode {
    /**
     * Pad with a constant color.
     *
     * @property [color] color to use.
     */
    public open class Fill(public val color: Color) : PaddingMode

    /**
     * Pad with black.
     */
    public object Black : Fill(Color.BLACK)
}
