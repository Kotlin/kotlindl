/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

import com.beust.klaxon.Json

internal data class KerasModel(
    val backend: String? = "tensorflow",
    @Json(serializeNull = false)
    val class_name: String? = null,
    @Json(serializeNull = false)
    val config: KerasModelConfig?,
    @Json(serializeNull = false)
    val keras_version: String? = null
)
