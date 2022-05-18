/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

import com.beust.klaxon.Json

internal data class KerasInitializerConfig(
    @Json(serializeNull = false)
    val distribution: String? = null,
    @Json(serializeNull = false)
    val maxval: Double? = null,
    @Json(serializeNull = false)
    val mean: Double? = null,
    @Json(serializeNull = false)
    val minval: Double? = null,
    @Json(serializeNull = false)
    val mode: String? = null,
    @Json(serializeNull = false)
    val scale: Double? = null,
    @Json(serializeNull = false)
    val seed: Int? = null,
    @Json(serializeNull = false)
    val stddev: Double? = null,
    @Json(serializeNull = false)
    val value: Double? = null,
    @Json(serializeNull = false)
    val gain: Double? = null,
    @Json(serializeNull = false)
    val p1: Double? = null,
    @Json(serializeNull = false)
    val p2: Double? = null
)
