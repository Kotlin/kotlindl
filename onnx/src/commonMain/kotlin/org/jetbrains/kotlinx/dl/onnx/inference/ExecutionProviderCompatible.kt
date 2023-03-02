/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference

import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.CPU

/**
 * Interface for a different kinds of ONNX models which support different execution providers.
 */
public interface ExecutionProviderCompatible : AutoCloseable {
    /**
     * Initialize the model with the specified executions providers.
     */
    public fun initializeWith(vararg executionProviders: ExecutionProvider = arrayOf(CPU(true)))
}
