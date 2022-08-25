/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessing.bitmap

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation

/**
 * This image preprocessor defines the Resize operation.
 *
 * Resize operations creates new [Bitmap] with the following sizes:
 *  - resizedWidth = [outputWidth]
 *  - resizedHeight = [outputHeight]
 *
 *
 * @property [outputWidth] The output width.
 * @property [outputHeight] The output height.
 */
public class Resize(
    private var outputWidth: Int = 100,
    private var outputHeight: Int = 100,
) : Operation<Bitmap, Bitmap> {
    override fun apply(input: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(input, outputWidth, outputHeight, true)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return when (inputShape.rank()) {
            2 -> TensorShape(outputWidth.toLong(), outputHeight.toLong())
            3 -> TensorShape(outputWidth.toLong(), outputHeight.toLong(), inputShape[2])
            else -> throw IllegalArgumentException("Input shape must expected to be 2D or 3D")
        }
    }
}
