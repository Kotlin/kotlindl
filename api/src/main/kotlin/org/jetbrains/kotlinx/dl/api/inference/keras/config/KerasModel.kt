/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

internal data class KerasModel(
    val backend: String? = "tensorflow",
    val class_name: String? = "Sequential",
    val config: SequentialConfig?,
    val keras_version: String? = "2.2.4-tf"
)
