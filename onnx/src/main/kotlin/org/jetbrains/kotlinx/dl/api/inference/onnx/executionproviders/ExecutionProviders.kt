package org.jetbrains.kotlinx.dl.api.inference.onnx.executionproviders

import ai.onnxruntime.OrtProvider

public object ExecutionProviders {
    public sealed class ExecutionProvider(public val internalProviderId: OrtProvider) {
        public data class CPU(public val useBFCArenaAllocator: Boolean = true) : ExecutionProvider(OrtProvider.CPU) {
            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (javaClass != other?.javaClass) return false
                if (!super.equals(other)) return false

                other as CPU

                if (useBFCArenaAllocator != other.useBFCArenaAllocator) return false

                return true
            }

            override fun hashCode(): Int {
                var result = super.hashCode()
                result = 31 * result + useBFCArenaAllocator.hashCode()
                return result
            }
        }

        public data class CUDA(public val deviceId: Int = 0) : ExecutionProvider(OrtProvider.CUDA) {
            override fun equals(other: Any?): Boolean {
                if (this === other) return true
                if (javaClass != other?.javaClass) return false
                if (!super.equals(other)) return false

                other as CUDA

                if (deviceId != other.deviceId) return false

                return true
            }

            override fun hashCode(): Int {
                var result = super.hashCode()
                result = 31 * result + deviceId
                return result
            }
        }

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (javaClass != other?.javaClass) return false

            other as ExecutionProvider

            if (internalProviderId != other.internalProviderId) return false

            return true
        }

        override fun hashCode(): Int {
            return internalProviderId.hashCode()
        }
    }
}