/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline

/**
 * Convenience functions for executing custom logic after applying [Operation].
 * Could be useful for debugging purposes.
 */
public fun <I, O> Operation<I, O>.onResult(block: (O) -> Unit): Operation<I, O> {
    return PreprocessingPipeline(this, object : Operation<O, O> {
        override fun apply(input: O): O {
            block(input)
            return input
        }

        override fun getOutputShape(inputShape: TensorShape): TensorShape = inputShape
    })
}

/**
 * Applies provided [operation] to the preprocessing pipeline.
 */
public fun <I, M, O> Operation<I, M>.call(operation: Operation<M, O>): Operation<I, O> {
    return PreprocessingPipeline(this, operation)
}
