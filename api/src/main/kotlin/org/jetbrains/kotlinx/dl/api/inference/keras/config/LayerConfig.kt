/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config

internal data class LayerConfig(
    val activation: String? = null,
    val activity_regularizer: ActivityRegularizer? = null,
    var batch_input_shape: List<Int?>? = null,
    val bias_constraint: Any? = null,
    val bias_initializer: KerasInitializer? = null,
    val bias_regularizer: KerasRegularizer? = null,
    val data_format: String? = null,
    val dilation_rate: List<Int>? = null,
    val dtype: String? = null,
    val filters: Int? = null,
    val kernel_constraint: Any? = null,
    val kernel_initializer: KerasInitializer? = null,
    val kernel_regularizer: KerasRegularizer? = null,
    val kernel_size: List<Int>? = null,
    val name: String? = null,
    val padding: String? = null,
    val pool_size: List<Int>? = null,
    val strides: List<Int>? = null,
    val trainable: Boolean? = true,
    val units: Int? = null,
    val use_bias: Boolean? = null
)