/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * This class represents the batch of data in the [Dataset].
 * @param [x] Data observations.
 * @param [y] Labels.
 * @param [elementShape] Shape of the data elements.
 */
public data class DataBatch(val x: Array<FloatArray>, val elementShape: TensorShape, val y: FloatArray) {
    /**
     * Number of rows in the batch.
     */
    public val size: Int get() = x.size

    /**
     * Shape of this [DataBatch].
     */
    public val shape: TensorShape get() = TensorShape(size.toLong(), *elementShape.dims())

    init {
        check(x.size == y.size) {
            "Number of data elements in the batch (${x.size}) is not the same as the number of labels (${y.size})."
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other == null || this::class != other::class) return false

        other as DataBatch

        return x.contentDeepEquals(other.x) && y.contentEquals(other.y)
    }

    override fun hashCode(): Int {
        var result = x.contentDeepHashCode()
        result = 31 * result + y.contentHashCode()
        return result
    }
}
