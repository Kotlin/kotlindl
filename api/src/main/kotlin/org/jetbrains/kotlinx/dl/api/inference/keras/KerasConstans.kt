/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

// Keras layers
internal const val LAYER_CONV2D: String = "Conv2D"
internal const val LAYER_DEPTHWISE_CONV2D: String = "DepthwiseConv2D"
internal const val LAYER_SEPARABLE_CONV2D: String = "SeparableConv2D"
internal const val LAYER_DENSE: String = "Dense"
internal const val LAYER_MAX_POOLING_2D: String = "MaxPooling2D"
internal const val LAYER_AVG_POOLING_2D: String = "AvgPooling2D"
internal const val LAYER_AVGERAGE_POOLING_2D: String = "AveragePooling2D"
internal const val LAYER_RESCALING: String = "Rescaling"
internal const val LAYER_NORMALIZATION: String = "Normalization"
internal const val LAYER_FLATTEN: String = "Flatten"
internal const val LAYER_RESHAPE: String = "Reshape"
internal const val LAYER_ZERO_PADDING_2D = "ZeroPadding2D"
internal const val LAYER_CROPPING_2D = "Cropping2D"
internal const val LAYER_BATCH_NORM: String = "BatchNormalization"
internal const val LAYER_ACTIVATION: String = "Activation"
internal const val LAYER_RELU: String = "ReLU"
internal const val LAYER_LSTM: String = "LSTM"
internal const val LAYER_DROPOUT: String = "Dropout"
internal const val LAYER_ADD: String = "Add"
internal const val LAYER_MULTIPLY: String = "Multiply"
internal const val LAYER_SUBTRACT: String = "Subtract"
internal const val LAYER_AVERAGE: String = "Average"
internal const val LAYER_MAXIMUM: String = "Maximum"
internal const val LAYER_MINIMUM: String = "Minimum"
internal const val LAYER_CONCATENATE: String = "Concatenate"
internal const val LAYER_GLOBAL_AVG_POOLING_2D: String = "GlobalAveragePooling2D"

// Keras data types
internal const val DATATYPE_FLOAT32: String = "float32"

// Keras Initializers
internal const val INITIALIZER_GLOROT_UNIFORM: String = "GlorotUniform"
internal const val INITIALIZER_GLOROT_NORMAL: String = "GlorotNormal"
internal const val INITIALIZER_HE_UNIFORM: String = "HeUniform"
internal const val INITIALIZER_HE_NORMAL: String = "HeNormal"
internal const val INITIALIZER_LECUN_UNIFORM: String = "LeCunUniform"
internal const val INITIALIZER_LECUN_NORMAL: String = "LeCunNormal"
internal const val INITIALIZER_ZEROS: String = "Zeros"
internal const val INITIALIZER_ONES: String = "Ones"
internal const val INITIALIZER_RANDOM_NORMAL: String = "RandomNormal"
internal const val INITIALIZER_RANDOM_UNIFORM: String = "RandomUniform"
internal const val INITIALIZER_TRUNCATED_NORMAL: String = "TruncatedNormal"
internal const val INITIALIZER_CONSTANT: String = "Constant"
internal const val INITIALIZER_VARIANCE_SCALING: String = "VarianceScaling"

// Keras activations
internal const val ACTIVATION_RELU: String = "relu"
internal const val ACTIVATION_SIGMOID: String = "sigmoid"
internal const val ACTIVATION_SOFTMAX: String = "softmax"
internal const val ACTIVATION_LINEAR: String = "linear"
internal const val ACTIVATION_SOFTPLUS: String = "softplus"
internal const val ACTIVATION_SOFTSIGN: String = "softsign"
internal const val ACTIVATION_RELU6: String = "relu6"
internal const val ACTIVATION_TANH: String = "tanh"
internal const val ACTIVATION_HARD_SIGMOID: String = "hard_sigmoid"
internal const val ACTIVATION_SWISH: String = "swish"
internal const val ACTIVATION_ELU: String = "elu"
internal const val ACTIVATION_SELU: String = "selu"
internal const val ACTIVATION_LOG_SOFTMAX: String = "log_softmax"
internal const val ACTIVATION_EXP: String = "exponential"

// Layer settings
internal const val CHANNELS_LAST: String = "channels_last"
internal const val CHANNELS_FIRST = "channels_first"
internal const val PADDING_SAME: String = "same"
