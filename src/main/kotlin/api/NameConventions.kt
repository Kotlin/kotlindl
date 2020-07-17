package api

import api.keras.layers.Layer

fun defaultLayerName(cnt: Int) = "layer_$cnt"

fun <T : Number> defaultActivationName(layer: Layer<T>) = "Activation_${layer.name}"

fun defaultAssignOpName(name: String) = "Assign_$name"

fun defaultInitializerOpName(name: String) = "Init_$name"