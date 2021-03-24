/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.api.extension.set3D

public class Cropping(public val top: Int, public val bottom: Int, public val left: Int, public val right: Int) :
    ImagePreprocessor {
    override fun apply(image: FloatArray, inputShape: ImageShape): Pair<FloatArray, ImageShape> {
        // TODO: check input parameters with inputShape on logic
        val croppedImageShape =
            ImageShape(
                width = inputShape.width - left - right,
                height = inputShape.height - top - bottom,
                channels = inputShape.channels
            )
        val croppedImage = FloatArray(croppedImageShape.numberOfElements.toInt())

        for (i in 0 until croppedImageShape.width.toInt()) {
            for (j in 0 until croppedImageShape.height.toInt()) {
                for (k in 0 until croppedImageShape.channels.toInt()) {
                    croppedImage.set3D(
                        i, j, k, croppedImageShape.width.toInt(), croppedImageShape.channels.toInt(), image.get3D(
                            i + top, j + left, k,
                            inputShape.width.toInt(),
                            inputShape.channels.toInt()
                        )
                    )
                }
            }
        }

        return Pair(croppedImage, croppedImageShape)
    }
}
