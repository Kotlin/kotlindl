/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.util

import org.jetbrains.kotlinx.dl.api.core.layer.Layer

/** Default activation name in TensorFlow graph, based on [layer]'s name. */
internal fun defaultActivationName(layer: Layer) = "Activation_${layer.name}"

/** Default Assign op name in TensorFlow graph, based on variable's name. */
internal fun defaultAssignOpName(name: String) = "Assign_$name"

/** Default Initializer op name in TensorFlow graph, based on variable's name. */
internal fun defaultInitializerOpName(name: String) = "Init_$name"

/** Default optimizer variable name in TensorFlow graph, based on variable's name. */
internal fun defaultOptimizerVariableName(name: String) = "optimizer_$name"

/** Default Conv2d bias variable name in TensorFlow graph, based on variable's name. */
internal fun conv2dBiasVarName(name: String) = name + "_" + "conv2d_bias"

/** Default Conv2d kernel variable name in TensorFlow graph, based on variable's name. */
internal fun conv2dKernelVarName(name: String) = name + "_" + "conv2d_kernel"

/** Default DepthwiseConv2d bias variable name in TensorFlow graph, based on variable's name. */
internal fun depthwiseConv2dBiasVarName(name: String) = name + "_" + "depthwise_conv2d_bias"

/** Default DepthwiseConv2d kernel variable name in TensorFlow graph, based on variable's name. */
internal fun depthwiseConv2dKernelVarName(name: String) = name + "_" + "depthwise_conv2d_kernel"

/** Default SeparableConv2d bias variable name in TensorFlow graph, based on variable's name. */
internal fun separableConv2dBiasVarName(name: String) = name + "_" + "separable_conv2d_bias"

/** Default SeparableConv2d depthwise kernel variable name in TensorFlow graph, based on variable's name. */
internal fun separableConv2dDepthwiseKernelVarName(name: String) = name + "_" + "separable_conv2d_depthwise_kernel"

/** Default SeparableConv2d pointwise kernel variable name in TensorFlow graph, based on variable's name. */
internal fun separableConv2dPointwiseKernelVarName(name: String) = name + "_" + "separable_conv2d_pointwise_kernel"

/** Default Dense bias variable name in TensorFlow graph, based on variable's name. */
internal fun denseBiasVarName(name: String) = name + "_" + "dense_bias"

/** Default Dense kernel variable name in TensorFlow graph, based on variable's name. */
internal fun denseKernelVarName(name: String) = name + "_" + "dense_kernel"

internal fun batchNormGammaVarName(name: String) = name + "_" + "batch_norm_gamma"

internal fun batchNormBetaVarName(name: String) = name + "_" + "batch_norm_beta"

internal fun batchNormMovingMeanVarName(name: String) = name + "_" + "batch_norm_moving_mean"

internal fun batchNormMovingVarianceVarName(name: String) = name + "_" + "batch_norm_moving_variance"
