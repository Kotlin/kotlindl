/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider


/**
 * Convenience extension functions for inference of ONNX models using different execution providers.
 */

/**
 * Extension function for explicitly specifying the execution provider for specific code block.
 * Follows [kotlin.use] semantics.
 */
public inline fun <M : ExecutionProviderCompatible, R> M.inferAndCloseUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    this.initializeWith(*providers)
    return this.use(block)
}

/**
 * Extension function for explicitly specifying the execution provider for specific code block.
 * Follows [kotlin.run] semantics.
 */
public inline fun <M : ExecutionProviderCompatible, R> M.inferUsing(
    vararg providers: ExecutionProvider,
    block: (M) -> R
): R {
    this.initializeWith(*providers)
    return this.run(block)
}
