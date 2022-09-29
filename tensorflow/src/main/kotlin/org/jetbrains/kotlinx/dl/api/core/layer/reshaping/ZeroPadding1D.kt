/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.tensorflow.Shape

/**
 * Zero-padding layer for 1D input (e.g. audio).
 * This layer can add zeros in the rows of the audio tensor
 *
 * __Input shape:__  3D tensor with shape `(batch_size, axis_to_pad, features)`.
 *
 * __Output shape:__ 3D tensor with shape `(batch_size, padded_axis, features)`.
 *
 * @property [padding] 2 numbers  interpreted as `(left_pad, right_pad)`.
 *
 * @since 0.3
 */
public class ZeroPadding1D : AbstractZeroPadding {
    public val padding: IntArray

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] symmetric padding applied to width(same on all sides)
     * @param [name] layer name
     */
    public constructor(
        padding: Int,
        name: String = ""
    ) : this(
        IntArray(2) { padding },
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] triple of padding values - [Pair.first] represents padding on left side
     * and [Pair.second] is padding on right side
     * @param [name] layer name
     */
    public constructor(
        padding: Pair<Int, Int>,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.second),
        name
    )

    /**
     * Constructs an instance of ZeroPadding1D layer
     * @param [padding] list of padding values. Size of list must be equal to 2. Those list values maps to
     * the following paddings:
     * padding[0] -> left padding,
     * padding[1] -> right padding,
     * @param [name] layer name
     */
    public constructor(padding: IntArray, name: String = "") : super(name) {
        require(padding.size == 2)
        this.padding = padding
    }

    override fun paddingArrayToTfFormat(inputShape: Shape): Array<IntArray> {
        return arrayOf(intArrayOf(0, 0), intArrayOf(padding[0], padding[1]), intArrayOf(0, 0))
    }

    override fun toString(): String {
        return "ZeroPadding1D(name = $name, padding=${padding.contentToString()}, hasActivation=$hasActivation)"
    }
}
