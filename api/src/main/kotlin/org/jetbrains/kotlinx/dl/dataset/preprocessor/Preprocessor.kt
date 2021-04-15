/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/**
 * Basic interface for the data preprocessing.
 *
 * It operates on [FloatArray].
 */
public interface Preprocessor {
    /**
     * Transforms [data] with [inputShape] to the new data with the same shape.
     *
     * @return Transformed data.
     */
    public fun apply(data: FloatArray, inputShape: ImageShape): FloatArray // move to shape
}




