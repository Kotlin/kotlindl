package org.jetbrains.kotlinx.dl.api.inference.onnx

import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider
import org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders.ExecutionProvider.CPU

public interface ExecutionProviderCompatible {
    public fun initializeWith(vararg executionProviders: ExecutionProvider = arrayOf(CPU(true)))
}
