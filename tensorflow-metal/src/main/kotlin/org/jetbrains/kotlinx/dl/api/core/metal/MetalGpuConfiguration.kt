/*
 * Copyright 2026 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metal

import org.jetbrains.kotlinx.dl.api.core.GpuConfiguration

/**
 * A [GpuConfiguration] with the settings the Metal PluggableDevice needs.
 *
 * `allowSoftPlacement` defaults to `true` here, and should stay that way. The plugin implements no
 * Metal kernel for the `Assign` op, so the variables of a graph pinned hard to the GPU cannot be
 * placed and the session fails with a colocation error before running a single step. With soft
 * placement those ops fall back to the CPU while convolutions, matrix multiplies, activations and
 * the optimizer update stay on Metal.
 *
 * ```
 * MetalAcceleration.enable()
 * val model = Sequential.of(layers, gpuConfiguration = metalGpuConfiguration())
 * ```
 *
 * @param allowGrowth whether the device may grow its memory allocation on demand.
 * @param allowSoftPlacement see above; leaving this `false` will break training on Metal.
 */
public fun metalGpuConfiguration(
    allowGrowth: Boolean? = true,
    allowSoftPlacement: Boolean = true,
): GpuConfiguration = GpuConfiguration(
    allowGrowth = allowGrowth,
    allowSoftPlacement = allowSoftPlacement,
)
