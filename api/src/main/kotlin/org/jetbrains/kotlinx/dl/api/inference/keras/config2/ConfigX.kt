/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.config2

data class ConfigX(
    val activation: String?,
    val activity_regularizer: Any?,
    val axis: List<Int>?,
    val batch_input_shape: List<Any>?,
    val beta_constraint: Any?,
    val beta_initializer: BetaInitializer?,
    val beta_regularizer: Any?,
    val bias_constraint: Any?,
    val bias_initializer: BiasInitializer?,
    val bias_regularizer: Any?,
    val center: Boolean?,
    val data_format: String?,
    val depth_multiplier: Int?,
    val depthwise_constraint: Any?,
    val depthwise_initializer: DepthwiseInitializer?,
    val depthwise_regularizer: Any?,
    val dilation_rate: List<Int>?,
    val dtype: String?,
    val epsilon: Double?,
    val filters: Int?,
    val gamma_constraint: Any?,
    val gamma_initializer: GammaInitializer?,
    val gamma_regularizer: Any?,
    val kernel_constraint: Any?,
    val kernel_initializer: KernelInitializer?,
    val kernel_regularizer: Any?,
    val kernel_size: List<Int>?,
    val max_value: Double?,
    val momentum: Double?,
    val moving_mean_initializer: MovingMeanInitializer?,
    val moving_variance_initializer: MovingVarianceInitializer?,
    val name: String?,
    val negative_slope: Double?,
    val noise_shape: Any?,
    val padding: Any?,
    val rate: Double?,
    val scale: Boolean?,
    val seed: Any?,
    val sparse: Boolean?,
    val strides: List<Int>?,
    val target_shape: List<Int>?,
    val threshold: Double?,
    val trainable: Boolean?,
    val use_bias: Boolean?
)
