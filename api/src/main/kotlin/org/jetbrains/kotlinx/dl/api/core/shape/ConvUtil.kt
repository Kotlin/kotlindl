/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.shape

import org.jetbrains.kotlinx.dl.api.core.layer.twodim.ConvPadding

/**
 * Calculates output length.
 */
internal fun convOutputLength(
    inputLength: Long,
    filterSize: Int,
    padding: ConvPadding,
    stride: Int,
    dilation: Int = 1
): Long {
    val dilatedFilterSize = filterSize + (filterSize - 1) * (dilation - 1)
    val outputLength = when (padding) {
        ConvPadding.SAME -> inputLength
        ConvPadding.VALID -> inputLength - dilatedFilterSize + 1
        ConvPadding.FULL -> inputLength + dilatedFilterSize - 1
    }
    return ((outputLength + stride - 1).toFloat() / stride).toLong()
}