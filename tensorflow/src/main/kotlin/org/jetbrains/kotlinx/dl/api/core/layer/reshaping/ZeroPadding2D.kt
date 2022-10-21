/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Shape

/**
 * Zero-padding layer for 2D input (e.g. picture).
 * This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.
 * @property [padding] 4 numbers  interpreted as `(top_pad, bottom_pad, left_pad, right_pad)`.
 */
public class ZeroPadding2D : AbstractZeroPadding {
    public val padding: IntArray
    private val dataFormat: String

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param [padding] symmetric padding applied to width and height (same on all sides)
     * @param [dataFormat] one of [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST]
     * or [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param [name] layer name
     */
    public constructor(
        padding: Int,
        dataFormat: String?,
        name: String = ""
    ) : this(
        IntArray(4) { padding },
        dataFormat,
        name
    )

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param [padding] pair of padding values - [Pair.first] represents vertical padding (applied to top and
     * bottom of image, and [Pair.second] is horizontal padding (left and right sides)
     * @param [dataFormat] one of [CHANNELS_FIRST]
     * or [CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param [name] layer name
     */
    public constructor(
        padding: Pair<Int, Int>,
        dataFormat: String?,
        name: String = ""
    ) : this(
        intArrayOf(padding.first, padding.first, padding.second, padding.second),
        dataFormat,
        name
    )

    /**
     * Constructs an instance of ZeroPadding2D layer
     * @param [padding] list of padding values. Size of list must be equal to 4. Those list values maps to
     * the following paddings:
     * padding[0] -> top padding,
     * padding[1] -> bottom padding,
     * padding[2] -> left padding,
     * padding[3] -> right padding
     * @param [dataFormat] one of [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST]
     * or [org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST], depending on dataFormat of
     * input to this layer
     * @param [name] layer name
     */
    public constructor(padding: IntArray, dataFormat: String?, name: String = "") : super(name) {
        require(padding.size == 4)
        this.padding = padding
        this.dataFormat = dataFormat ?: CHANNELS_LAST
    }

    override fun paddingArrayToTfFormat(inputShape: Shape): Array<IntArray> {
        val paddingFirstDim: IntArray
        val paddingSecondDim: IntArray

        when (padding.size) {
            4 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[1])
                paddingSecondDim = intArrayOf(padding[2], padding[3])
            }
            2 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[1], padding[1])
            }
            1 -> {
                paddingFirstDim = intArrayOf(padding[0], padding[0])
                paddingSecondDim = intArrayOf(padding[0], padding[0])
            }
            else -> throw IllegalArgumentException("Invalid padding argument at layer $name.")
        }
        return paddingArraysToInputShape(inputShape, paddingFirstDim, paddingSecondDim)
    }

    private fun paddingArraysToInputShape(
        inputShape: Shape,
        paddingFirstDim: IntArray,
        paddingSecondDim: IntArray
    ): Array<IntArray> {
        return when (inputShape.numDimensions()) {
            4 -> {
                if (dataFormat == CHANNELS_FIRST) {
                    arrayOf(intArrayOf(0, 0), intArrayOf(0, 0), paddingFirstDim, paddingSecondDim)
                } else {
                    arrayOf(intArrayOf(0, 0), paddingFirstDim, paddingSecondDim, intArrayOf(0, 0))
                }
            }
            3 -> {
                if (dataFormat == CHANNELS_FIRST) {
                    arrayOf(intArrayOf(0, 0), paddingFirstDim, paddingSecondDim)
                } else {
                    arrayOf(paddingFirstDim, paddingSecondDim, intArrayOf(0, 0))
                }
            }
            2 -> {
                arrayOf(paddingFirstDim, paddingSecondDim)
            }
            else -> throw IllegalArgumentException("Invalid input shape $inputShape.")
        }
    }

    override fun toString(): String {
        return "ZeroPadding2D(name = $name, padding=${padding.contentToString()}, hasActivation=$hasActivation)"
    }
}
