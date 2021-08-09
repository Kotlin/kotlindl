/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/**
 * This wrapper preprocessor applies the passed custom preprocessor to be used in Preprocessing DSL.
 *
 * @property [customPreprocessor] Custom preprocessor.
 */
public class Sharpen(public var customPreprocessor: Preprocessor?) : Preprocessor {
    override fun apply(data: FloatArray, inputShape: ImageShape): FloatArray {
        return customPreprocessor?.apply(data, inputShape) ?: data
    }
}
