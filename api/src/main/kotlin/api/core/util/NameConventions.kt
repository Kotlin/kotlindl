package api.core.util

import api.core.layer.Layer

/** Default activation name in TensorFlow graph, based on [layer]'s name. */
fun defaultActivationName(layer: Layer) = "Activation_${layer.name}"

/** Default Assign op name in TensorFlow graph, based on variable's name. */
fun defaultAssignOpName(name: String) = "Assign_$name"

/** Default Initializer op name in TensorFlow graph, based on variable's name. */
fun defaultInitializerOpName(name: String) = "Init_$name"

/** Default optimizer variable name in TensorFlow graph, based on variable's name. */
fun defaultOptimizerVariableName(name: String) = "optimizer_$name"

/** Default Conv2d bias variable name in TensorFlow graph, based on variable's name. */
fun conv2dBiasVarName(name: String) = name + "_" + "conv2d_bias"

/** Default Conv2d kernel variable name in TensorFlow graph, based on variable's name. */
fun conv2dKernelVarName(name: String) = name + "_" + "conv2d_kernel"

/** Default Dense bias variable name in TensorFlow graph, based on variable's name. */
fun denseBiasVarName(name: String) = name + "_" + "dense_bias"

/** Default Dense kernel variable name in TensorFlow graph, based on variable's name. */
fun denseKernelVarName(name: String) = name + "_" + "dense_kernel"