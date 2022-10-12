/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.bitmap

import android.graphics.Bitmap
import android.graphics.Matrix
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * This image preprocessor defines the Rotate operation.
 *
 * It rotates the input [Bitmap] by [degrees].
 *
 * @property [degrees] The rotation angle.
 */
public class Rotate(
    public var degrees: Float = 0.0f,
) : Operation<Bitmap, Bitmap> {
    override fun apply(input: Bitmap): Bitmap {
        if (degrees == 0f) return input

        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(input, 0, 0, input.width, input.height, matrix, true)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
