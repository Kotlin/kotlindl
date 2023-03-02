/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.onnx.inference.executionproviders

import ai.onnxruntime.OrtProvider
import ai.onnxruntime.OrtSession
import ai.onnxruntime.providers.NNAPIFlags
import org.jetbrains.kotlinx.dl.onnx.inference.executionproviders.ExecutionProvider.*
import java.util.*

/**
 * These are classes representing the supported ONNXRuntime execution providers for KotlinDL.
 * The supported providers are:
 *  - [CPU] (default)
 *  - [CUDA] (could be used if the CUDA runtime is installed)
 *  - [NNAPI] (could be used on Android if the NNAPI runtime is supported)
 *
 * Internally, the [OrtProvider] enum is used to indicate the provider.
 *
 * @param [internalProviderId] the [OrtProvider] enum value
 */
public sealed class ExecutionProvider(public val internalProviderId: OrtProvider) {
    /**
     *  Default CPU execution provider.
     *  Available on all platforms.
     *
     *  @param useBFCArenaAllocator If true, the CPU provider will use BFC arena allocator.
     *  @see [OrtProvider.CPU]
     */
    public data class CPU(public val useBFCArenaAllocator: Boolean = true) : ExecutionProvider(OrtProvider.CPU) {
        override fun addOptionsTo(sessionOptions: OrtSession.SessionOptions) {
            sessionOptions.addCPU(useBFCArenaAllocator)
        }
    }

    /**
     *  CUDA execution provider.
     *  Available only on platforms with Nvidia gpu and CUDA runtime installed.
     *
     *  @param deviceId The device ID to use.
     *  @see [OrtProvider.CUDA]
     */
    public data class CUDA(public val deviceId: Int = 0) : ExecutionProvider(OrtProvider.CUDA) {
        override fun addOptionsTo(sessionOptions: OrtSession.SessionOptions) {
            sessionOptions.addCUDA(deviceId)
        }
    }

    /**
     *  NNAPI execution provider.
     *  Available only on Android.
     *
     *  @param flags An NNAPI flags to modify the behavior of the NNAPI execution provider.
     *  @see [OrtProvider.NNAPI]
     *  @see <a href=https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html>NNAPI documentation</a>.
     */
    public data class NNAPI(public val flags: Set<NNAPIFlags> = emptySet()) : ExecutionProvider(OrtProvider.NNAPI) {
        override fun addOptionsTo(sessionOptions: OrtSession.SessionOptions) {
            val internalNNAPIFlags = EnumSet.noneOf(NNAPIFlags::class.java)
            flags.let { internalNNAPIFlags.addAll(it) }
            sessionOptions.addNnapi(internalNNAPIFlags)
        }
    }

    /**
     * Adds execution provider options to the [OrtSession.SessionOptions].
     */
    public open fun addOptionsTo(sessionOptions: OrtSession.SessionOptions): Unit = Unit
}
