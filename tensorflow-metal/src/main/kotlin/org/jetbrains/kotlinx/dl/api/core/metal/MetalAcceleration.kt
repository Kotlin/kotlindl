/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import org.jetbrains.kotlinx.dl.api.core.metal.internal.PluggableDeviceLoader
import org.tensorflow.DeviceSpec
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.op.Ops
import java.nio.file.Path

/**
 * Registers Apple's [tensorflow-metal](https://developer.apple.com/metal/tensorflow-plugin/)
 * PluggableDevice so KotlinDL's TensorFlow models run on the Apple GPU instead of the CPU.
 *
 * Opt-in and process-wide: [enable] registers the device with the TensorFlow runtime, after which
 * models built with a [org.jetbrains.kotlinx.dl.api.core.GpuConfiguration] that pins them to the GPU
 * will use it. Registration cannot be undone within a process.
 *
 * ### Requirements
 *
 * * macOS on Apple Silicon (`aarch64`). No other platform is supported by the plugin.
 * * JDK 22 or later — this module calls the Foreign Function & Memory API.
 * * A copy of `libmetal_plugin.dylib`, which is **not** bundled. See [MetalPlugin] for how it is
 *   located and how to obtain it without installing Python.
 * * `--enable-native-access=ALL-UNNAMED` on the JVM command line. Loading a native library through
 *   the FFM API is a restricted operation: without the flag the JVM prints a warning, and a future
 *   release will refuse the call outright.
 *
 * ### Soft placement is required
 *
 * The plugin provides no Metal kernel for the `Assign` op, so a graph pinned hard to the GPU fails
 * to place its variables. Use [metalGpuConfiguration] (or set `allowSoftPlacement = true` yourself)
 * so those ops fall back to the CPU while the compute stays on Metal.
 *
 * ### Coverage
 *
 * Conv2D, depthwise convolution, 2D pooling, dense layers, the common activations and the
 * SGD/Adam/Momentum/RMSProp/Adadelta/Adagrad optimizers all run on Metal. 3D convolution and
 * pooling, the `Elu`/`Selu`/`Softsign`/`Softplus` activations, the `AdaGradDA`/`Ftrl` optimizers and
 * gradient clipping have no Metal kernel and silently fall back to the CPU.
 *
 * Example:
 * ```
 * when (val status = MetalAcceleration.enable()) {
 *     is MetalStatus.Enabled   -> println("Running on the Apple GPU")
 *     is MetalStatus.Unavailable -> println("Falling back to CPU: ${status.reason}")
 * }
 *
 * val model = Sequential.of(layers, gpuConfiguration = metalGpuConfiguration())
 * ```
 */
public object MetalAcceleration {

    private var enabled: MetalStatus.Enabled? = null

    /**
     * Whether the current JVM and machine could host the plugin at all: macOS on Apple Silicon,
     * running on JDK 22 or later.
     *
     * This is a check of the environment only — it does not imply that the plugin itself is
     * present. [enable] is what determines that.
     */
    @get:JvmStatic
    public val isSupportedPlatform: Boolean
        get() = PlatformSupport.describe() == null

    /** Whether [enable] has already succeeded in this process. */
    @get:JvmStatic
    public val isEnabled: Boolean
        get() = enabled != null

    /**
     * Registers the Metal PluggableDevice, and verifies that an operation can actually be placed on
     * it.
     *
     * Safe to call more than once. A **successful** registration is remembered and returned by later
     * calls, because the device cannot be registered twice or unregistered. Failures are deliberately
     * *not* remembered, so a caller that supplies a wrong path, or has not installed the plugin yet,
     * can fix the cause and call again.
     *
     * @param pluginPath explicit location of `libmetal_plugin.dylib`, or of a `tensorflow_metal`
     *   wheel to take it from. When `null`, [MetalPlugin.locate] searches the documented locations.
     * @return [MetalStatus.Enabled] if a Metal device is registered and usable, otherwise
     *   [MetalStatus.Unavailable] describing why. This method does not throw for an absent or
     *   unusable plugin — callers are expected to degrade to CPU.
     */
    @JvmStatic
    @JvmOverloads
    @Synchronized
    public fun enable(pluginPath: Path? = null): MetalStatus {
        enabled?.let { return it }

        val result = runCatching { register(pluginPath) }
            .getOrElse { MetalStatus.Unavailable("Failed to register the Metal device: ${it.message}") }

        if (result is MetalStatus.Enabled) enabled = result
        return result
    }

    private fun register(pluginPath: Path?): MetalStatus {
        PlatformSupport.describe()?.let { return MetalStatus.Unavailable(it) }

        val plugin = pluginPath?.let { MetalPlugin.resolve(it) }
            ?: MetalPlugin.locate()
            ?: return MetalStatus.Unavailable(MetalPlugin.notFoundMessage())

        val loaded = PluggableDeviceLoader.load(plugin)
        if (loaded is PluggableDeviceLoader.Result.Failed) {
            return MetalStatus.Unavailable(loaded.reason)
        }

        // Registration reporting success is not the same as a usable device: confirm by placing a
        // real operation on the GPU. Without this a misconfigured plugin would be reported as
        // working and every model would fail later, far from the cause.
        return runCatching { placeOperationOnGpu() }
            .fold(
                onSuccess = { MetalStatus.Enabled(plugin) },
                onFailure = {
                    MetalStatus.Unavailable(
                        "The Metal plugin loaded from $plugin but no operation could be placed on " +
                                "the GPU: ${it.message?.lineSequence()?.firstOrNull()}"
                    )
                }
            )
    }

    private fun placeOperationOnGpu() {
        Graph().use { graph ->
            val tf = Ops.create(graph)
                .withDevice(DeviceSpec.newBuilder().deviceType(DeviceSpec.DeviceType.GPU).build())
            val product = tf.linalg.matMul(
                tf.constant(arrayOf(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f))),
                tf.constant(arrayOf(floatArrayOf(1f, 0f), floatArrayOf(0f, 1f)))
            )
            Session(graph).use { session ->
                session.runner().fetch(product).run().use { /* forces placement and execution */ }
            }
        }
    }
}

/** Outcome of [MetalAcceleration.enable]. */
public sealed class MetalStatus {
    /** A Metal device is registered and an operation was successfully executed on it. */
    public data class Enabled(
        /** The plugin library that was loaded. */
        public val pluginPath: Path
    ) : MetalStatus()

    /** Metal is not available; [reason] explains why, and the caller should stay on the CPU. */
    public data class Unavailable(public val reason: String) : MetalStatus()
}

internal object PlatformSupport {
    /** Returns null when the platform is supported, otherwise a human-readable reason why not. */
    fun describe(): String? {
        val os = System.getProperty("os.name").orEmpty()
        if (!os.lowercase().contains("mac")) {
            return "Apple Metal requires macOS, but this JVM reports os.name=\"$os\"."
        }
        val arch = System.getProperty("os.arch").orEmpty()
        if (arch != "aarch64" && arch != "arm64") {
            return "tensorflow-metal is published for Apple Silicon only, but this JVM reports " +
                    "os.arch=\"$arch\". Intel Macs have no Metal plugin."
        }
        // Backstop only. This class is compiled to Java 22 bytecode, so on an older JVM it fails
        // to load with UnsupportedClassVersionError long before this check could run. It is kept so
        // the requirement is stated in one place with the others.
        val feature = Runtime.version().feature()
        if (feature < 22) {
            return "This module needs the Foreign Function & Memory API, which is final in JDK 22; " +
                    "this JVM is Java $feature."
        }
        return null
    }
}
