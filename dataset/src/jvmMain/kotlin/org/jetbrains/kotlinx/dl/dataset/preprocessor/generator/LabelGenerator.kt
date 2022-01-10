/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.generator

import java.io.File

/** A parent interface for all label generators. */
public interface LabelGenerator {
    /**
     * Returns a label for the provided [file].
     */
    public fun getLabel(file: File): Float

    public companion object {
        internal fun LabelGenerator.prepareY(sources: Array<File>): FloatArray {
            return FloatArray(sources.size) { getLabel(sources[it]) }
        }
    }
}
