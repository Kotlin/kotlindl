/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal.internal

import org.tensorflow.TensorFlow
import java.lang.foreign.Arena
import java.lang.foreign.FunctionDescriptor
import java.lang.foreign.Linker
import java.lang.foreign.MemorySegment
import java.lang.foreign.SymbolLookup
import java.lang.foreign.ValueLayout
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption

/**
 * Calls TensorFlow's `TF_LoadPluggableDeviceLibrary` through the Foreign Function & Memory API.
 *
 * TensorFlow Java 1.0.0 binds 853 C API entry points but not this one — it was added upstream in
 * [tensorflow/java#605](https://github.com/tensorflow/java/pull/605) and first shipped in 1.1.0,
 * which KotlinDL does not use because it drops the `macosx-x86_64` native. The symbol is exported by
 * the shipped `libtensorflow_cc` dylib regardless, so it can be called directly.
 *
 * ### Why a shim is needed
 *
 * `libmetal_plugin.dylib` declares `@rpath/_pywrap_tensorflow_internal.so` — the CPython extension
 * module of the Python TensorFlow build — as a hard dynamic dependency. Every symbol it imports is
 * already exported by the pair of dylibs TensorFlow Java ships, so what is missing is only a library
 * carrying that *name*. [SHIM_RESOURCE] is a stub that carries it and re-exports both.
 *
 * The shim is loaded **before** the plugin. macOS uses a two-level namespace, so dyld satisfies the
 * plugin's dependency from the already-loaded image and never searches the filesystem for it. That
 * is why the shim can live in a temporary directory and nothing has to be written next to the
 * TensorFlow natives.
 */
internal object PluggableDeviceLoader {

    private const val SHIM_RESOURCE =
        "org/jetbrains/kotlinx/dl/api/core/metal/macosx-arm64/_pywrap_tensorflow_internal.so"

    private const val SHIM_FILE_NAME = "_pywrap_tensorflow_internal.so"

    /** Held for the process lifetime: unloading either library would break the registered device. */
    private val arena = Arena.ofShared()

    private var shimLookup: SymbolLookup? = null

    sealed interface Result {
        data object Loaded : Result
        data class Failed(val reason: String) : Result
    }

    fun load(pluginPath: Path): Result {
        // Force TensorFlow's natives into the process first. The shim re-exports them by install
        // name, so they must already be mapped for its own load to resolve.
        TensorFlow.version()

        if (shimLookup == null) {
            val shim = loadShim()
            if (shim is Result.Failed) return shim
        }
        val lookup = shimLookup
            ?: return Result.Failed("The Metal shim loaded but no symbol lookup was retained.")

        val linker = Linker.nativeLinker()
        val address = ValueLayout.ADDRESS

        // invokeWithArguments rather than invokeExact: the latter is signature-polymorphic, which
        // Kotlin does not reliably compile. These calls happen once, so the reflective path costs
        // nothing that matters.
        fun handleOf(name: String, descriptor: FunctionDescriptor) =
            lookup.find(name).map { linker.downcallHandle(it, descriptor) }.orElse(null)

        val newStatus = handleOf("TF_NewStatus", FunctionDescriptor.of(address))
        val loadLibrary = handleOf(
            "TF_LoadPluggableDeviceLibrary", FunctionDescriptor.of(address, address, address)
        )
        val getCode = handleOf("TF_GetCode", FunctionDescriptor.of(ValueLayout.JAVA_INT, address))
        val message = handleOf("TF_Message", FunctionDescriptor.of(address, address))

        if (newStatus == null || loadLibrary == null || getCode == null || message == null) {
            return Result.Failed(
                "The TensorFlow C API symbols needed to register a PluggableDevice are not reachable " +
                        "through the shim. This usually means the shipped TensorFlow natives changed layout."
            )
        }

        Arena.ofConfined().use { call ->
            val status = newStatus.invokeWithArguments() as MemorySegment
            val path = call.allocateFrom(pluginPath.toAbsolutePath().toString())

            loadLibrary.invokeWithArguments(path, status)

            val code = getCode.invokeWithArguments(status) as Int
            if (code != 0) {
                val pointer = message.invokeWithArguments(status) as MemorySegment
                val text = if (pointer == MemorySegment.NULL) "no detail available"
                else pointer.reinterpret(Long.MAX_VALUE).getString(0).lineSequence().first()
                return Result.Failed("TensorFlow rejected $pluginPath (status $code): $text")
            }
        }
        return Result.Loaded
    }

    /** Extracts the bundled shim to a temporary file and loads it. */
    private fun loadShim(): Result {
        val resource = javaClass.classLoader.getResourceAsStream(SHIM_RESOURCE)
            ?: return Result.Failed(
                "The Metal shim is missing from this artifact. It can only be linked on macOS arm64, " +
                        "so a jar built on another platform will not contain it."
            )

        val target = try {
            val directory = Files.createTempDirectory("kotlindl-metal")
            directory.toFile().deleteOnExit()
            val file = directory.resolve(SHIM_FILE_NAME)
            resource.use { input ->
                Files.copy(input, file, StandardCopyOption.REPLACE_EXISTING)
            }
            file.toFile().deleteOnExit()
            file
        } catch (e: Exception) {
            return Result.Failed("Could not unpack the Metal shim: ${e.message}")
        }

        return try {
            shimLookup = SymbolLookup.libraryLookup(target, arena)
            Result.Loaded
        } catch (e: IllegalArgumentException) {
            Result.Failed("Could not load the Metal shim from $target: ${e.message}")
        }
    }
}
