/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config2

data class Config(
    val input_layers: List<List<Any>>?,
    val layers: List<Layer>?,
    val name: String?,
    val output_layers: List<List<Any>>?
)
