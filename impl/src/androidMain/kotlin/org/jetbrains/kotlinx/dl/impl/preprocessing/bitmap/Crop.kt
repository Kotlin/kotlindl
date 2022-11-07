/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.bitmap

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.ColorSpace
import android.graphics.Rect
import android.os.Build
import androidx.annotation.RequiresApi
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * This class defines the Crop [Operation].
 *
 * Crop operation crops the given image at the specified location and size.
 * If the cropped region does not fit into the input image, the image is padded first and then cropped.
 *
 * @property [x]      cropped region top-left corner x-coordinate
 * @property [y]      cropped region top-left corner y-coordinate
 * @property [width]  cropped region width
 * @property [height] cropped region height
 */
@RequiresApi(Build.VERSION_CODES.O)
public class Crop(
    public var x: Int = 0,
    public var y: Int = 0,
    public var width: Int = 0,
    public var height: Int = 0,
) : Operation<Bitmap, Bitmap> {

    override fun apply(input: Bitmap): Bitmap {
        if (width < 0 || height < 0) {
            throw IllegalArgumentException(
                "Negative crop width and height are not allowed. " +
                        "Current parameters: width = $width, height = $height"
            )
        }
        if (x == 0 && y == 0 && width == input.width && height == input.height) {
            return input
        }
        if (x >= 0 && y >= 0 && x + width <= input.width && y + height <= input.height) {
            return Bitmap.createBitmap(input, x, y, width, height)
        }
        val output = Bitmap.createBitmap(
            width, height, input.config, input.hasAlpha(),
            input.colorSpace ?: ColorSpace.get(ColorSpace.Named.SRGB)
        )
        val inputRect = Rect(
            x.coerceAtLeast(0),
            y.coerceAtLeast(0),
            (x + width).coerceAtMost(input.width),
            (y + height).coerceAtMost(input.height)
        )
        val outputRect = Rect(inputRect).apply {
            offsetTo((-x).coerceAtLeast(0), (-y).coerceAtLeast(0))
        }
        Canvas(output).drawBitmap(input, inputRect, outputRect, null)
        return output
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return Resize.createOutputImageShape(inputShape, width, height)
    }
}