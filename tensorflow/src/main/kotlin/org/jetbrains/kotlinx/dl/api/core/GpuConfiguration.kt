package org.jetbrains.kotlinx.dl.api.core

import org.tensorflow.framework.ConfigProto
import org.tensorflow.framework.GPUOptions

/**
 * Represents the configuration for the GPU in a TensorFlow session.
 *
 * @property allowGrowth Whether to allow dynamic GPU memory growth. Defaults to null.
 * @property allocatorType The type of memory allocator to use on the GPU. Defaults to null.
 * @property forceGpuCompatible Whether to force all tensors to be GPU compatible. Defaults to null.
 * @property deferredDeletionBytes The number of bytes of GPU memory that can be freed asynchronously. Defaults to null.
 * @property perProcessGpuMemoryFraction The fraction of the overall GPU memory that each process is allowed to use. Defaults to null.
 * @property pollingActiveDelayUsecs The delay in microseconds between GPU memory polling cycles when memory is actively used. Defaults to null.
 * @property pollingInactiveDelayMsecs The delay in milliseconds between GPU memory polling cycles when memory is inactive. Defaults to null.
 */
public class GpuConfiguration(
    public val allowGrowth: Boolean? = null,
    public val allocatorType: String? = null,
    public val forceGpuCompatible: Boolean? = null,
    public val deferredDeletionBytes: Long? = null,
    public val perProcessGpuMemoryFraction: Double? = null,
    public val pollingActiveDelayUsecs: Int? = null,
    public val pollingInactiveDelayMsecs: Int? = null,
) {
    public fun toTensorFlowSessionConfig(): ByteArray? {
        val gpuOptions: GPUOptions.Builder = GPUOptions.newBuilder()
        if (allowGrowth != null)
            gpuOptions.setAllowGrowth(allowGrowth)
        if (allocatorType != null)
            gpuOptions.setAllocatorType(allocatorType)
        if (forceGpuCompatible != null)
            gpuOptions.setForceGpuCompatible(forceGpuCompatible)
        if (deferredDeletionBytes != null)
            gpuOptions.setDeferredDeletionBytes(deferredDeletionBytes)
        if (perProcessGpuMemoryFraction != null)
            gpuOptions.setPerProcessGpuMemoryFraction(perProcessGpuMemoryFraction)
        if (pollingActiveDelayUsecs != null)
            gpuOptions.setPollingActiveDelayUsecs(pollingActiveDelayUsecs)
        if (pollingInactiveDelayMsecs != null)
            gpuOptions.setPollingInactiveDelayMsecs(pollingInactiveDelayMsecs)

        val config: ConfigProto = ConfigProto.newBuilder()
            .setGpuOptions(gpuOptions)
            .build()

        return config.toByteArray()
    }
}