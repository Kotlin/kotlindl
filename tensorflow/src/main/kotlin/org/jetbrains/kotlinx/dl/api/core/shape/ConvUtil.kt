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
    if (outputPaddingEnd == null || outputPaddingStart == null) {
        // https://github.com/keras-team/keras/blob/cff8cc93305d1c4a54385fb623fe895dafa0845c/keras/utils/conv_utils.py#L185
        return when (padding) {
            ConvPadding.VALID -> inputLength * stride + max(dilatedFilterSize - stride, 0)
            ConvPadding.SAME -> inputLength * stride
            ConvPadding.FULL -> inputLength * stride - (stride + dilatedFilterSize - 2)
        }
    }
    val totalPadding = convTransposePadding(padding, outputPaddingStart, outputPaddingEnd, filterSize, dilation).sum()
    return (inputLength - 1) * stride + dilatedFilterSize + totalPadding
}

private fun convTransposePadding(
    padding: ConvPadding,
    outputPaddingStart: Int,
    outputPaddingEnd: Int,
    dilatedKernelSize: Int
): List<Int> {
    // https://github.com/keras-team/keras/blob/cff8cc93305d1c4a54385fb623fe895dafa0845c/keras/utils/conv_utils.py#L194
    val automaticPadding = when (padding) {
        ConvPadding.VALID -> 0
        ConvPadding.SAME -> dilatedKernelSize / 2
        ConvPadding.FULL -> dilatedKernelSize - 1
    }
    return listOf(outputPaddingStart - automaticPadding, outputPaddingEnd - automaticPadding)
}

internal fun convTransposePadding(
    padding: ConvPadding,
    outputPaddingStart: Int,
    outputPaddingEnd: Int,
    filterSize: Int,
    dilation: Int
): List<Int> {
    val dilatedKernelSize = dilatedFilterSize(filterSize, dilation)
    return convTransposePadding(padding, outputPaddingStart, outputPaddingEnd, dilatedKernelSize).map { max(0, it) }
}