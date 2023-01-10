/*
 * Copyright 2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

/**
 * Provides utility methods for converting inference result to the common data types.
 * @param [R] inference result type.
 * @see InferenceModel
 */
public interface InferenceResultConverter<R> {
    /**
     * Returns the output at [index] as a [FloatArray].
     */
    public fun getFloatArray(result: R, index: Int): FloatArray

    /**
     * Returns the output at [index] as a [LongArray].
     */
    public fun getLongArray(result: R, index: Int): LongArray
}