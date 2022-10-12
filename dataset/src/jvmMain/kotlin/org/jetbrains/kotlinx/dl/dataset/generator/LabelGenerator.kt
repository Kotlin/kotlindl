/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.generator

/** A parent interface for all label generators. */
public interface LabelGenerator<D> {
    /**
     * Returns a label for the provided [dataSource].
     */
    public fun getLabel(dataSource: D): Float

    public companion object {
        internal fun <D> LabelGenerator<D>.prepareY(sources: Array<D>): FloatArray {
            return FloatArray(sources.size) { getLabel(sources[it]) }
        }
    }
}
