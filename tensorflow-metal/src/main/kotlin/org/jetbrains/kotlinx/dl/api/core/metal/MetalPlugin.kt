/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import java.nio.file.Files
import java.nio.file.Path
import java.util.zip.ZipFile
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.name

/**
 * Locates `libmetal_plugin.dylib`, the Metal PluggableDevice implementation.
 *
 * KotlinDL does not bundle it: it is Apple's binary, published only inside the `tensorflow-metal`
 * CPython wheel, under its own licence. That wheel is an ordinary zip — **no Python installation is
 * needed**. Download it from PyPI and either point KotlinDL at the `.whl` directly, or extract it:
 *
 * ```
 * unzip -j tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl \
 *       tensorflow-plugins/libmetal_plugin.dylib -d ~/.kotlindl/metal
 * ```
 *
 * Version 1.2.0 is recommended: it is the only release whose symbol requirements are fully
 * satisfied by both TensorFlow Java 1.0.0 and 1.1.0.
 *
 * Search order used by [locate]:
 *
 * 1. the `kotlindl.metal.plugin` system property,
 * 2. the `KOTLINDL_METAL_PLUGIN` environment variable,
 * 3. `~/.kotlindl/metal/libmetal_plugin.dylib`.
 *
 * Each may name the dylib itself, a directory containing it, or a `tensorflow_metal` wheel.
 */
public object MetalPlugin {

    /** System property that may point at the plugin, a directory holding it, or a wheel. */
    public const val PLUGIN_PROPERTY: String = "kotlindl.metal.plugin"

    /** Environment variable equivalent of [PLUGIN_PROPERTY]. */
    public const val PLUGIN_ENV_VAR: String = "KOTLINDL_METAL_PLUGIN"

    /** Library file name inside the wheel and on disk. */
    public const val LIBRARY_NAME: String = "libmetal_plugin.dylib"

    private const val WHEEL_ENTRY_PREFIX = "tensorflow-plugins/"

    private val defaultLocation: Path
        get() = Path.of(System.getProperty("user.home"), ".kotlindl", "metal", LIBRARY_NAME)

    /**
     * Searches the documented locations and returns a usable plugin library, or `null` if none was
     * found. Wheels are extracted to a cache directory next to [defaultLocation].
     */
    @JvmStatic
    public fun locate(): Path? {
        val candidates = listOfNotNull(
            System.getProperty(PLUGIN_PROPERTY),
            System.getenv(PLUGIN_ENV_VAR)
        ).map { Path.of(it) } + defaultLocation

        return candidates.firstNotNullOfOrNull { candidate ->
            runCatching { resolve(candidate) }.getOrNull()
        }
    }

    /**
     * Interprets [location] as the plugin library, a directory containing it, or a
     * `tensorflow-metal` wheel to extract it from, and returns the resulting dylib.
     *
     * @throws IllegalArgumentException if nothing usable is found at that location.
     */
    @JvmStatic
    public fun resolve(location: Path): Path {
        require(location.exists()) { "No such file or directory: $location" }

        val resolved = when {
            location.isDirectory() -> location.resolve(LIBRARY_NAME)
            location.extension.equals("whl", ignoreCase = true) -> extractFromWheel(location)
            else -> location
        }

        require(resolved.exists()) { "$LIBRARY_NAME not found at $resolved" }
        require(resolved.name == LIBRARY_NAME) {
            "Expected $LIBRARY_NAME but got ${resolved.name}; refusing to load an unexpected library."
        }
        return resolved
    }

    /** Message describing how to obtain the plugin, used when the search comes up empty. */
    @JvmStatic
    public fun notFoundMessage(): String =
        "$LIBRARY_NAME was not found. Set -D$PLUGIN_PROPERTY=<path>, the $PLUGIN_ENV_VAR " +
                "environment variable, or place it at $defaultLocation. The path may be the dylib, a " +
                "directory containing it, or a tensorflow_metal wheel (a plain zip -- Python is not " +
                "required). Download the wheel from https://pypi.org/project/tensorflow-metal/."

    /**
     * Extracts `tensorflow-plugins/libmetal_plugin.dylib` from a wheel into a cache directory keyed
     * by the wheel's file name, so repeated runs do not re-extract.
     */
    private fun extractFromWheel(wheel: Path): Path {
        val target = defaultLocation.parent.resolve(wheel.name.removeSuffix(".whl")).resolve(LIBRARY_NAME)
        if (target.exists()) return target

        ZipFile(wheel.toFile()).use { zip ->
            val entry = zip.entries().asSequence().firstOrNull {
                !it.isDirectory && it.name.startsWith(WHEEL_ENTRY_PREFIX) && it.name.endsWith(LIBRARY_NAME)
            } ?: throw IllegalArgumentException(
                "$wheel does not contain $WHEEL_ENTRY_PREFIX$LIBRARY_NAME; is it a tensorflow-metal wheel?"
            )

            target.parent.createDirectories()
            zip.getInputStream(entry).use { input ->
                Files.newOutputStream(target).use(input::copyTo)
            }
        }
        return target
    }
}
