/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape

/**
 * And interface for loading data from the provided data source.
 * @param D data source type
 */
public interface DataLoader<D> {
    /**
     * Load the data from the specified [dataSource].
     */
    public fun load(dataSource: D): FloatData

    public companion object {
        internal fun <D> DataLoader<D>.prepareX(sources: Array<D>): Pair<Array<FloatArray>, TensorShape> {
            return prepareX(sources, 0, sources.size)
        }

        internal fun <D> DataLoader<D>.prepareX(
            src: Array<D>,
            start: Int,
            length: Int
        ): Pair<Array<FloatArray>, TensorShape> {
            if (length == 0) return emptyArray<FloatArray>() to TensorShape()

            val shapes = mutableSetOf<TensorShape>()
            val array = Array(length) { index ->
                val (floats, shape) = load(src[start + index])
                shapes.add(shape)
                floats
            }
            require(shapes.size == 1) {
                "Dataset elements should have the same shape. Current shapes: $shapes"
            }
            return array to shapes.single()
        }
    }
}