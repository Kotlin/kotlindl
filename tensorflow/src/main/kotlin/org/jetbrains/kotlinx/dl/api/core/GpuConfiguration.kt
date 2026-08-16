package org.jetbrains.kotlinx.dl.api.core

import org.tensorflow.proto.ConfigProto
import org.tensorflow.proto.GPUOptions

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
 * @property allowSoftPlacement Whether TensorFlow may place an operation on another device when the
 *   requested one has no kernel for it, instead of failing. Defaults to null (TensorFlow's own
 *   default, which is off). Required by accelerators that implement only part of the op set — the
 *   Apple Metal PluggableDevice, for instance, has no kernel for the `Assign` op, so a graph pinned
 *   hard to the GPU cannot place its variables.
 */
public class GpuConfiguration(
    public val allowGrowth: Boolean? = null,
    public val allocatorType: String? = null,
    public val forceGpuCompatible: Boolean? = null,
    public val deferredDeletionBytes: Long? = null,
    public val perProcessGpuMemoryFraction: Double? = null,
    public val pollingActiveDelayUsecs: Int? = null,
    public val pollingInactiveDelayMsecs: Int? = null,
    public val allowSoftPlacement: Boolean? = null,
) {
    /**
     * Builds the session configuration.
     *
     * Returns a [ConfigProto] message rather than its serialized bytes: TensorFlow Java 1.x takes
     * the parsed proto in the [org.tensorflow.Session] constructor.
     */
    public fun toTensorFlowSessionConfig(): ConfigProto {
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

        val config = ConfigProto.newBuilder().setGpuOptions(gpuOptions)
        if (allowSoftPlacement != null)
            config.setAllowSoftPlacement(allowSoftPlacement)

        return config.build()
    }
}