/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation

/**
 * The aim of this class is to provide common functionality for all [Operation]s that can be applied to Pair<FloatArray, TensorShape>
 * and simplify the implementation of a new [Operation]s.
 */
public abstract class FloatArrayOperation : Operation<Pair<FloatArray, TensorShape>, Pair<FloatArray, TensorShape>> {
    protected abstract fun applyImpl(data: FloatArray, shape: TensorShape): FloatArray

    override fun apply(input: Pair<FloatArray, TensorShape>): Pair<FloatArray, TensorShape> {
        val (data, shape) = input
        return applyImpl(data, shape) to getOutputShape(shape)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
}
