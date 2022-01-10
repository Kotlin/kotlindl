/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import java.io.File

/**
 * And interface for loading data from [File].
 */
public interface DataLoader {
    /**
     * Load the data from the specified [file].
     */
    public fun load(file: File): Pair<FloatArray, TensorShape>

    public companion object {
        internal fun DataLoader.prepareX(sources: Array<File>): Array<FloatArray> {
            return Array(sources.size) { load(sources[it]).first }
        }
    }
}