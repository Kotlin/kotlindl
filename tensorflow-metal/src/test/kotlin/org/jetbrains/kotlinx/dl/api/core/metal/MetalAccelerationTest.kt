/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIf
import org.junit.jupiter.api.io.TempDir
import java.nio.file.Path
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.outputStream
import kotlin.io.path.readBytes
import kotlin.io.path.writeText

/**
 * Tests that run everywhere. The parts that need a real Metal device live in
 * [MetalDeviceTest], which is skipped unless the plugin is actually present.
 */
class MetalAccelerationTest {

    @Test
    fun platformSupportMatchesTheRunningJvm() {
        val reason = PlatformSupport.describe()
        val isAppleSilicon = System.getProperty("os.name").lowercase().contains("mac") &&
                System.getProperty("os.arch") in setOf("aarch64", "arm64")

        if (isAppleSilicon && Runtime.version().feature() >= 22) {
            assertNull(reason, "Expected this platform to be supported, but got: $reason")
            assertTrue(MetalAcceleration.isSupportedPlatform)
        } else {
            assertNotNull(reason, "Expected an explanation of why Metal is unsupported here.")
            assertFalse(MetalAcceleration.isSupportedPlatform)
        }
    }

    /**
     * The point of the module is that it never forces a caller onto a failure path: with no plugin
     * installed, [MetalAcceleration.enable] must report the problem rather than throw.
     */
    @Test
    fun enablingWithAMissingPluginReportsRatherThanThrows() {
        val missing = Path.of("/definitely/not/a/real/path/libmetal_plugin.dylib")

        val status = runCatching { MetalAcceleration.enable(missing) }
        assertTrue(status.isSuccess, "enable() must not throw for an absent plugin: ${status.exceptionOrNull()}")

        // On a machine where a previous test already enabled Metal the memoised Enabled result is
        // returned instead, which is also correct behaviour.
        val value = status.getOrThrow()
        assertTrue(
            value is MetalStatus.Unavailable || value is MetalStatus.Enabled,
            "Unexpected status: $value"
        )
    }

    @Test
    fun notFoundMessageExplainsEverySupportedLocation() {
        val message = MetalPlugin.notFoundMessage()
        assertTrue(message.contains(MetalPlugin.PLUGIN_PROPERTY), "should mention the system property")
        assertTrue(message.contains(MetalPlugin.PLUGIN_ENV_VAR), "should mention the environment variable")
        assertTrue(message.contains("wheel"), "should explain the wheel route")
        assertTrue(message.contains("Python is not"), "should make clear Python is unnecessary")
    }

    @Test
    fun resolveRejectsAPathThatDoesNotExist() {
        val error = runCatching { MetalPlugin.resolve(Path.of("/no/such/file.dylib")) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException, "Expected IllegalArgumentException, got $error")
    }

    @Test
    fun resolveRejectsALibraryWithAnUnexpectedName(@TempDir dir: Path) {
        val impostor = dir.resolve("libsomething_else.dylib")
        impostor.writeText("not really a library")

        val error = runCatching { MetalPlugin.resolve(impostor) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException, "Expected the name check to reject it, got $error")
    }

    @Test
    fun resolveFindsTheLibraryInsideADirectory(@TempDir dir: Path) {
        val library = dir.resolve(MetalPlugin.LIBRARY_NAME)
        library.writeText("stand-in")

        assertEquals(library, MetalPlugin.resolve(dir))
    }

    /**
     * A `tensorflow_metal` wheel is a plain zip, so the dylib can be taken from it without Python.
     * This builds a miniature wheel to prove the extraction path works.
     */
    @Test
    fun resolveExtractsTheLibraryFromAWheel(@TempDir dir: Path) {
        val payload = "pretend Mach-O contents".toByteArray()
        val wheel = dir.resolve("tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl")
        ZipOutputStream(wheel.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("tensorflow-plugins/${MetalPlugin.LIBRARY_NAME}"))
            zip.write(payload)
            zip.closeEntry()
            zip.putNextEntry(ZipEntry("tensorflow_metal-1.2.0.dist-info/METADATA"))
            zip.write("Name: tensorflow-metal".toByteArray())
            zip.closeEntry()
        }

        val extracted = MetalPlugin.resolve(wheel)
        assertTrue(extracted.exists(), "the dylib should have been written out")
        assertEquals(MetalPlugin.LIBRARY_NAME, extracted.fileName.toString())
        assertTrue(payload.contentEquals(extracted.readBytes()), "extracted contents should match the wheel entry")

        // Extraction is cached, so a second call must return the same file without re-reading.
        assertEquals(extracted, MetalPlugin.resolve(wheel))
    }

    @Test
    fun resolveRejectsAZipThatIsNotAMetalWheel(@TempDir dir: Path) {
        val wheel = dir.resolve("something-1.0.0-py3-none-any.whl")
        ZipOutputStream(wheel.outputStream()).use { zip ->
            zip.putNextEntry(ZipEntry("some/other/file.txt"))
            zip.write("hello".toByteArray())
            zip.closeEntry()
        }

        val error = runCatching { MetalPlugin.resolve(wheel) }.exceptionOrNull()
        assertTrue(error is IllegalArgumentException, "Expected a clear rejection, got $error")
        assertTrue(
            error!!.message!!.contains("tensorflow-metal"),
            "The message should say what kind of file was expected: ${error.message}"
        )
    }

    @Test
    fun metalConfigurationEnablesSoftPlacementByDefault() {
        val config = metalGpuConfiguration()
        assertEquals(true, config.allowSoftPlacement, "soft placement is mandatory for Metal")
        assertTrue(config.toTensorFlowSessionConfig().allowSoftPlacement)
    }

    /**
     * Soft placement must be off unless asked for, so this change cannot alter the behaviour of
     * existing models that do not opt in.
     */
    @Test
    fun defaultGpuConfigurationLeavesSoftPlacementOff() {
        val config = org.jetbrains.kotlinx.dl.api.core.GpuConfiguration()
        assertNull(config.allowSoftPlacement)
        assertFalse(config.toTensorFlowSessionConfig().allowSoftPlacement)
    }

    /** The shim is what makes the plugin loadable, so its absence should be caught at build time. */
    @Test
    @EnabledIf("isAppleSilicon")
    fun shimIsPackagedOnAppleSilicon() {
        val shim = javaClass.classLoader.getResource(
            "org/jetbrains/kotlinx/dl/api/core/metal/macosx-arm64/_pywrap_tensorflow_internal.so"
        )
        assertNotNull(shim, "The Metal shim should have been linked into this build by :buildMetalShim.")
    }

    @Suppress("unused")
    fun isAppleSilicon(): Boolean =
        System.getProperty("os.name").lowercase().contains("mac") &&
                System.getProperty("os.arch") in setOf("aarch64", "arm64")
}
