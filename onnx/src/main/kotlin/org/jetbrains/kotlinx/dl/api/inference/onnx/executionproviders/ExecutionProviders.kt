package org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders

import ai.onnxruntime.OrtProvider

/**
 * These are classes representing the supported ONNXRuntime execution providers for KotlinDL.
 * The supported providers are:
 *  - [CPU] (default)
 *  - [CUDA] (could be used if the CUDA runtime is installed)
 * Internally, the [OrtProvider] enum is used to indicate the provider.
 */
public sealed class ExecutionProvider(public val internalProviderId: OrtProvider) {
    /**
     *  Default CPU execution provider.
     *
     *  @param useBFCArenaAllocator If true, the CPU provider will use BFC arena allocator.
     *  @see [OrtProvider.CPU]
     */
    public data class CPU(public val useBFCArenaAllocator: Boolean = true) : ExecutionProvider(OrtProvider.CPU)

    /**
     *  CUDA execution provider.
     *
     *  @param deviceId The device ID to use.
     *  @see [OrtProvider.CUDA]
     */
    public data class CUDA(public val deviceId: Int = 0) : ExecutionProvider(OrtProvider.CUDA)
}
