/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

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

    public companion object {
        internal fun ImageShape.toTensorShape(): TensorShape {
            return TensorShape(width!!, height!!, channels!!)
        }
    }
}

public fun TensorShape.toImageShape(): ImageShape {
    return when (this.rank()) {
        2 -> ImageShape(this[0], this[1], 1)
        3 -> ImageShape(this[0], this[1], this[2])
        else -> throw IllegalArgumentException("Tensor shape must be 2D or 3D to be converted to ImageShape")
    }
}
