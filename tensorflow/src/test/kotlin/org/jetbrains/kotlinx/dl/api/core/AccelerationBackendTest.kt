/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledOnOs
import org.junit.jupiter.api.condition.OS
import org.tensorflow.DeviceSpec
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.TensorFlow

/**
 * Pins down which acceleration backends *this module on its own* can reach: none. On the classpath
 * of `:tensorflow` alone, TensorFlow runs on the CPU.
 *
 * Apple Metal is reachable, but only through the optional `:tensorflow-metal` artifact
 * (`kotlin-deeplearning-tensorflow-metal`), which targets JDK 22+ and calls
 * `TF_LoadPluggableDeviceLibrary` via the Foreign Function & Memory API. That module is deliberately
 * *not* a dependency here, which is exactly what these tests assert: nothing registers a device
 * behind a caller's back, and a plain `:tensorflow` user gets unchanged CPU behaviour.
 *
 * Note the distinction these tests draw. TensorFlow Java 1.0.0 exposes no *Java binding* for
 * pluggable devices, but the shipped native does export the symbol -- which is why the optional
 * module can reach it without new native code. A failure here therefore means one of two things:
 * a TensorFlow Java upgrade added the binding (it landed upstream in 1.1.0), or something put the
 * Metal plugin on this module's classpath. Neither is a bug to silence; both mean the GPU notes in
 * README.md need revisiting.
 */
class AccelerationBackendTest {

    /**
     * [TensorFlow] offers `loadLibrary` (custom op kernels) and `registerFilesystemPlugin`
     * (filesystem plugins) but no binding for `TF_LoadPluggableDeviceLibrary`, which is why
     * `:tensorflow-metal` has to call that symbol through the FFM API rather than through the Java
     * API. If this ever fails, the binding has arrived and that FFM code can be deleted.
     */
    @Test
    fun tensorFlowJavaExposesNoPluggableDeviceLoader() {
        val loaderLikeMethods = TensorFlow::class.java.methods
            .map { it.name }
            .filter { it.contains("pluggable", ignoreCase = true) || it.contains("device", ignoreCase = true) }

        assertTrue(
            loaderLikeMethods.isEmpty(),
            "TensorFlow Java now exposes device/pluggable-device entry points $loaderLikeMethods. " +
                    "PluggableDevice support may have landed, so Metal acceleration on macOS should be " +
                    "re-evaluated along with the GPU notes in README.md."
        )
    }

    /**
     * The Metal plugin ships as a Python wheel, so its native library should never appear on the
     * JVM classpath. If it ever does, something is being wired up that these tests do not describe.
     */
    @Test
    fun noMetalPluginIsOnTheClasspath() {
        val loader = AccelerationBackendTest::class.java.classLoader
        val metalArtifacts = listOf(
            "libmetal_plugin.dylib",
            "tensorflow-plugins/libmetal_plugin.dylib",
            "org/tensorflow/native/metal_plugin"
        ).filter { loader.getResource(it) != null }

        assertTrue(
            metalArtifacts.isEmpty(),
            "Found Metal plugin artifacts on the classpath: $metalArtifacts."
        )
    }

    /**
     * TensorFlow Java models device placement with a fixed set of device types. Metal is not one of
     * them; it would surface as a [DeviceSpec.DeviceType.CUSTOM] device only if the plugin were
     * registered, which the previous tests establish it cannot be.
     */
    @Test
    fun deviceTypesDoNotIncludeMetal() {
        val deviceTypes = DeviceSpec.DeviceType.values().map { it.name }
        assertFalse(
            deviceTypes.any { it.contains("METAL", ignoreCase = true) },
            "DeviceSpec.DeviceType now contains a Metal entry: $deviceTypes."
        )
    }

    /**
     * The functional counterpart: with only `:tensorflow` on the classpath, nothing has registered
     * a device, so pinning an op to a GPU must fail. TensorFlow Java publishes a GPU native for
     * `linux-x86_64` only, so this assertion is scoped to macOS.
     *
     * The equivalent positive assertion — that placement *succeeds* once the plugin is registered —
     * lives in `MetalDeviceTest` in the `:tensorflow-metal` module.
     */
    @Test
    @EnabledOnOs(OS.MAC)
    fun gpuPlacementIsUnavailableOnMacOs() {
        val gpu = DeviceSpec.newBuilder().deviceType(DeviceSpec.DeviceType.GPU).build()

        val placement = runCatching {
            Graph().use { graph ->
                val tf = org.tensorflow.op.Ops.create(graph).withDevice(gpu)
                val constant = tf.constant(1.0f)
                Session(graph).use { session ->
                    session.runner().fetch(constant).run().use { /* forces placement */ }
                }
            }
        }

        assertFalse(
            placement.isSuccess,
            "A GPU device became available with only :tensorflow on the classpath. TensorFlow Java " +
                    "ships no macOS GPU native, and registering Metal is supposed to require an explicit " +
                    "MetalAcceleration.enable() call from :tensorflow-metal -- so something is enabling " +
                    "an accelerator implicitly. Revisit the GPU notes in README.md."
        )

        // Guard against the assertion above passing for an unrelated reason: the failure must be
        // TensorFlow refusing to place the op, not some incidental error in this test.
        val message = placement.exceptionOrNull()?.message.orEmpty()
        assertTrue(
            message.contains("device", ignoreCase = true),
            "Expected a device placement failure, but got: $message"
        )
    }
}
