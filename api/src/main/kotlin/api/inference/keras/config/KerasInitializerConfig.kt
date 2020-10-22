/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package api.inference.keras.config

internal data class KerasInitializerConfig(
    val distribution: String? = null,
    val maxval: Double? = null,
    val mean: Double? = null,
    val minval: Double? = null,
    val mode: String? = null,
    val scale: Double? = null,
    val seed: Int? = null,
    val stddev: Double? = null,
    val value: Int? = null
)