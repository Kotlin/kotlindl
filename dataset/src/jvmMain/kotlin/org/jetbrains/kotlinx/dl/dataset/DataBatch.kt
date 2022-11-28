/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

/**
 * This class represents the batch of data in the [Dataset].
 * @param [x] Data observations.
 * @param [y] Labels.
 * @param [size] Number of rows in batch.
 */
public data class DataBatch internal constructor(
    val x: Array<FloatArray>,
    val y: FloatArray,
    val size: Int
) {
    /**
     * Returns 2-dimensional shape of the data batch.
     */
    public fun shape(elementSize: Int): List<Long> = listOf(size.toLong(), elementSize.toLong())

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DataBatch

        if (!x.contentDeepEquals(other.x)) return false
        if (!y.contentEquals(other.y)) return false
        return size == other.size
    }

    override fun hashCode(): Int {
        var result = x.contentDeepHashCode()
        result = 31 * result + y.contentHashCode()
        result = 31 * result + size
        return result
    }
}
