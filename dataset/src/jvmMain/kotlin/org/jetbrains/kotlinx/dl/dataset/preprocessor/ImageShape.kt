/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape

/**
 * Helper class to keep widely used shape of image object presented as a 4D tensor
 * (batchSize = 1, [width], [height], [channels]).
 * `null` values are allowed for [width], [height] and [channels], indicating that the dimension size is unknown.
 *
 * @property [width]    image width
 * @property [height]   image height
 * @property [channels] number of channels in the image
 * */
public data class ImageShape(
    public val width: Long? = null,
    public val height: Long? = null,
    public val channels: Long? = null
) {
    // TODO: add the flag (channelsLast = true) + getter shape in format (1, HWC) or (1, CHW)

    /** Returns number of elements in a tensor with the given shape. */
    public val numberOfElements: Long
        get() = width!! * height!! * channels!!
}

/**
 * Convenience function to create an [ImageShape] from [TensorShape].
 */
public fun TensorShape.toImageShape(): ImageShape {
    val width = if (this[0] == -1L) null else this[0]
    val height = if (this[1] == -1L) null else this[1]
    val channels = if (this[2] == -1L) null else this[2]

    return when (this.rank()) {
        2 -> ImageShape(width, height, 1)
        3 -> ImageShape(width, height, channels)
        else -> throw IllegalArgumentException("Tensor shape must be 2D or 3D to be converted to ImageShape")
    }
}
