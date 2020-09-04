package api

import api.keras.layers.Layer

fun defaultActivationName(layer: Layer) = "Activation_${layer.name}"

fun defaultAssignOpName(name: String) = "Assign_$name"

fun defaultInitializerOpName(name: String) = "Init_$name"

fun defaultOptimizerVariableName(name: String) = "optimizer_$name"

fun conv2dBiasVarName(name: String) = name + "_" + "conv2d_bias"

fun conv2dKernelVarName(name: String) = name + "_" + "conv2d_kernel"

fun denseBiasVarName(name: String) = name + "_" + "dense_bias"

fun denseKernelVarName(name: String) = name + "_" + "dense_kernel"