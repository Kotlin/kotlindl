/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * The aim of this class is to provide common functionality for all [Operation]s that can be applied to FloatData
 * and simplify the implementation of a new [Operation]s.
 */
public abstract class FloatArrayOperation : Operation<FloatData, FloatData> {
    /**
     * Actual implementation of the [Operation] that should be applied to the [data].
     */
    protected abstract fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray

    override fun apply(input: FloatData): FloatData {
        val (data, shape) = input
        return applyImpl(data, shape) to getOutputShape(shape)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
}
