package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU

/**
 * Interface for a different kinds of ONNX models which support different execution providers.
 */
public interface ExecutionProviderCompatible : AutoCloseable {
    /**
     * Initialize the model with the specified executions providers.
     */
    public fun initializeWith(vararg executionProviders: ExecutionProvider = arrayOf(CPU(true)))
}
