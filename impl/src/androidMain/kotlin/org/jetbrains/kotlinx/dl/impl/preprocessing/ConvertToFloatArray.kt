/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.impl.preprocessing.TensorLayout.NCHW
import org.jetbrains.kotlinx.dl.impl.preprocessing.TensorLayout.NHWC
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.argB8888ToNCHWArray
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.argB8888ToNHWCArray


/**
 * Converts [Bitmap] to float array representation.
 * Only [Bitmap.Config.ARGB_8888] is supported.
 *
 * @param layout [TensorLayout] of the resulting array.
 */
public class ConvertToFloatArray(public var layout: TensorLayout = NCHW) :
    Operation<Bitmap, FloatData> {
    private val channels = 3
    override fun apply(input: Bitmap): FloatData {
        require(input.config == Bitmap.Config.ARGB_8888) { "Only ARGB_8888 bitmaps are supported currently" }

        val w = input.width
        val h = input.height

        val encodedPixels = IntArray(w * h)
        input.getPixels(encodedPixels, 0, w, 0, 0, w, h)

        val tensor = when (layout) {
            NCHW -> argB8888ToNCHWArray(encodedPixels, w, h, channels)
            NHWC -> argB8888ToNHWCArray(encodedPixels, w, h, channels)
        }

        val shape = when (layout) {
            NCHW -> TensorShape(channels.toLong(), h.toLong(), w.toLong())
            NHWC -> TensorShape(h.toLong(), w.toLong(), channels.toLong())
        }

        return tensor to shape
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return when (inputShape.rank()) {
            2, 3 -> when (layout) {
                NCHW -> TensorShape(channels.toLong(), inputShape[0], inputShape[1])
                NHWC -> TensorShape(inputShape[0], inputShape[1], channels.toLong())
            }

            else -> throw IllegalArgumentException("Input shape must be 1D or 2D")
        }
    }
}
