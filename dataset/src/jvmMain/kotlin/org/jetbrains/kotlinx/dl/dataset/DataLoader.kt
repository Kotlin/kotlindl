/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.floats

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
        internal fun <D> DataLoader<D>.prepareX(sources: Array<D>): Array<FloatArray> {
            return Array(sources.size) { load(sources[it]).floats }
        }
    }
}