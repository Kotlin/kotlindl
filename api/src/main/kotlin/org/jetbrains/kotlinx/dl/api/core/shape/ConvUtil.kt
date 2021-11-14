/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import java.lang.Integer.max

private fun dilatedFilterSize(filterSize: Int, dilation: Int): Int {
    return filterSize + (filterSize - 1) * (dilation - 1)
}

private fun ConvPadding.value(filterSize: Int): Int {
    return when (this) {
        ConvPadding.VALID -> 0
        ConvPadding.SAME -> filterSize - 1
        ConvPadding.FULL -> 2 * filterSize - 1
    }
}

/**
 * Calculates output length after applying convolution operation on a single axis.
 */
internal fun convOutputLength(
    inputLength: Long,
    filterSize: Int,
    padding: ConvPadding,
    stride: Int,
    dilation: Int = 1
): Long {
    val dilatedFilterSize = dilatedFilterSize(filterSize, dilation)
    val outputLength = inputLength - filterSize + 1 + padding.value(dilatedFilterSize)
    return ((outputLength + stride - 1).toFloat() / stride).toLong()
}

/**
 * Calculates output length after applying transposed convolution operation on a single axis.
 */
internal fun convTransposeOutputLength(
    inputLength: Long,
    filterSize: Int,
    padding: ConvPadding,
    outputPaddingStart: Int?,
    outputPaddingEnd: Int?,
    stride: Int,
    dilation: Int
): Long {
    val dilatedFilterSize = dilatedFilterSize(filterSize, dilation)
    val totalPadding = (outputPaddingStart ?: 0) + (outputPaddingEnd ?: 0) - padding.value(dilatedFilterSize)
    return (inputLength - 1) * stride + dilatedFilterSize + totalPadding
}

internal fun convTransposeSingleSidePadding(padding: ConvPadding,
                                            outputPadding: Int,
                                            filterSize: Int,
                                            dilation: Int
): Int {
    val dilatedKernelSize = dilatedFilterSize(filterSize, dilation)
    val automaticPadding = padding.value(dilatedKernelSize) / 2
    return max(0, outputPadding - automaticPadding)
}