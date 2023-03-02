/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline

/** Applies [Normalizing] preprocessor to the tensor to normalize it with given mean and std values. */
public fun <I> Operation<I, FloatData>.normalize(block: Normalizing.() -> Unit): Operation<I, FloatData> {
    return PreprocessingPipeline(this, Normalizing().apply(block))
}

/** Applies [Rescaling] preprocessor to the tensor to scale each value by a given coefficient. */
public fun <I> Operation<I, FloatData>.rescale(block: Rescaling.() -> Unit): Operation<I, FloatData> {
    return PreprocessingPipeline(this, Rescaling().apply(block))
}
