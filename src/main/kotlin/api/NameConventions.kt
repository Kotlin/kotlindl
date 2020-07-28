package api

import api.keras.layers.Layer

fun defaultLayerName(cnt: Int) = "layer_$cnt"

fun <T : Number> defaultActivationName(layer: Layer<T>) = "Activation_${layer.name}"

fun defaultAssignOpName(name: String) = "Assign_$name"

fun defaultInitializerOpName(name: String) = "Init_$name"

fun conv2dBiasVarName(name: String) = name + "_" + "conv2d_bias"

fun conv2dKernelVarName(name: String) = name + "_" + "conv2d_kernel"

fun depthwiseConv2dBiasVarName(name: String) = name + "_" + "depthwise_conv2d_bias"

fun depthwiseConv2dKernelVarName(name: String) = name + "_" + "depthwise_conv2d_kernel"

fun denseBiasVarName(name: String) = name + "_" + "dense_bias"

fun denseKernelVarName(name: String) = name + "_" + "dense_kernel"