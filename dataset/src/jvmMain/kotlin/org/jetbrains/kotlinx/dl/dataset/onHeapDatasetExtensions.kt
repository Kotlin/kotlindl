/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.tensorShape
import kotlin.math.roundToInt


/**
 * Creates [OnHeapDataset] string representation for part of data.
 */
public fun OnHeapDataset.partialToString(): String = buildStringRepr(x.partialToString(), y.partialToString())

/**
 * Creates [OnHeapDataset] string representation for full of data.
 */
public fun OnHeapDataset.fullToString(): String = buildStringRepr(x.contentDeepToString(), y.contentToString())

/**
 * Builds intermediate [OnHeapDataset] string representation.
 */
public fun OnHeapDataset.buildStringRepr(xString: String, yString: String): String =
    "OnHeapDataset(\nx ${x.tensorShape} =\n${xString},\ny [${y.size}] =\n${yString}\n)"


/**
 * Create String representation of `FloatArray` where only a part of the data is printed to String.
 *
 * @param maxSize max number of elements of array present in its string representation
 * @param lowPercent percent of data of [maxSize] to be printed from the beginning of array data.
 * Rest will be obtained from the tail of the array in order matching the order in array
 * @return string representation of [FloatArray] in format like
 * `[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ..., 9.0, 10.0]`
 */
private fun FloatArray.partialToString(maxSize: Int = 10, lowPercent: Double = 0.8): String {
    if (size <= maxSize) {
        return contentToString()
    }

    val lowCount = (lowPercent * maxSize).roundToInt()
    val upStart = size - maxSize - 1

    return generateSequence(0, Int::inc).map {
        when {
            it < lowCount -> this[it]
            it > lowCount -> this[upStart + it]
            else -> "..."
        }
    }.take(maxSize + 1).joinToString(prefix = "[", postfix = "]", separator = ", ")
}

/**
 * Create String representation of `Array<FloatArray>` where only a part of the data is printed to String.
 *
 * @param maxSize max number of elements of array present in its string representation
 * @param lowPercent percent of data of [maxSize] to be printed from the beginning of array data.
 * Rest will be obtained from the tail of the array in order matching the order in array
 * @return string representation of [FloatArray] in format like
 * `[[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, ..., 9.0, 10.0],
 *   [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, ..., 20.0, 21.0],
 *   [22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, ..., 31.0, 32.0],
 *   [33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, ..., 42.0, 43.0],
 *   [44.0, 45.0, 46.0, 47.0, 48.0, 49.0, 50.0, 51.0, ..., 53.0, 54.0],
 *   [55.0, 56.0, 57.0, 58.0, 59.0, 60.0, 61.0, 62.0, ..., 64.0, 65.0],
 *   [66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, ..., 75.0, 76.0],
 *   [77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 84.0, ..., 86.0, 87.0],
 *   ...,
 *   [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, ..., 108.0, 109.0],
 *   [110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, ..., 119.0, 120.0]]`
 */
private fun Array<FloatArray>.partialToString(maxSize: Int = 10, lowPercent: Double = 0.8): String {
    if (size <= maxSize) {
        return joinToString(prefix = "[", postfix = "]", separator = ",\n ") {
            it.partialToString(maxSize, lowPercent)
        }
    }

    val lowCount = (lowPercent * maxSize).roundToInt()
    val upStart = size - maxSize - 1

    return generateSequence(0, Int::inc).map {
        when {
            it < lowCount -> this[it].partialToString(maxSize, lowPercent)
            it > lowCount -> this[upStart + it].partialToString(maxSize, lowPercent)
            else -> "..."
        }
    }.take(maxSize + 1).joinToString(prefix = "[", postfix = "]", separator = ",\n ")
}
