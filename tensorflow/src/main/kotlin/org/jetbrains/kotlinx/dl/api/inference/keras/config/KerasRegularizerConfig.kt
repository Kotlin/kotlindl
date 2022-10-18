/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

import com.beust.klaxon.Json

internal data class KerasRegularizerConfig(
    @Json(serializeNull = false)
    val l1: Double? = null,
    @Json(serializeNull = false)
    val l2: Double? = null
)
