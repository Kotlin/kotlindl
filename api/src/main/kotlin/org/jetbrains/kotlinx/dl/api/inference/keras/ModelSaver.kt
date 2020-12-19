/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.ZeroPadding2D
import org.jetbrains.kotlinx.dl.api.inference.keras.config.*
import java.io.File

/**
 * Saves model description as json configuration file fully compatible with the Keras TensorFlow framework.
 *
 * @param jsonConfigFile File to write model configuration.
 * @param isKerasFullyCompatible If true, it generates fully Keras-compatible configuration.
 */
public fun Sequential.saveModelConfiguration(jsonConfigFile: File, isKerasFullyCompatible: Boolean = false) {
    val kerasLayers = mutableListOf<KerasLayer>()
    this.layers.forEach {
        run {
            val layer = convertToKerasLayer(it, isKerasFullyCompatible)
            kerasLayers.add(layer)
        }
    }

    val inputShape = this.inputLayer.packedDims.map { it.toInt() }

    (kerasLayers.first().config as LayerConfig).batch_input_shape =
        listOf(null, inputShape[0], inputShape[1], inputShape[2])

    val config = SequentialConfig(name = "", layers = kerasLayers)
    val sequentialConfig = KerasSequentialModel(config = config)

    val jsonString2 = Klaxon().toJsonString(sequentialConfig)

    jsonConfigFile.writeText(jsonString2, Charsets.UTF_8)
}

private fun convertToKerasLayer(layer: Layer, isKerasFullyCompatible: Boolean): KerasLayer {
    return when (layer::class) {
        Conv2D::class -> createKerasConv2D(layer as Conv2D, isKerasFullyCompatible)
        Flatten::class -> createKerasFlatten(layer as Flatten)
        MaxPool2D::class -> createKerasMaxPooling2D(layer as MaxPool2D)
        AvgPool2D::class -> createKerasAvgPooling2D(layer as AvgPool2D)
        Dense::class -> createKerasDense(layer as Dense, isKerasFullyCompatible)
        ZeroPadding2D::class -> createKerasZeroPadding2D(layer as ZeroPadding2D)
        else -> throw IllegalStateException("${layer.name} is not supported yet!")
    }
}

private fun createKerasDense(layer: Dense, isKerasFullyCompatible: Boolean): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        units = layer.outputSize,
        name = layer.name,
        use_bias = true,
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible)
    )
    return KerasLayer(class_name = LAYER_DENSE, config = configX)
}

private fun convertToKerasInitializer(initializer: Initializer, isKerasFullyCompatible: Boolean): KerasInitializer? {
    val className: String
    val config: KerasInitializerConfig
    if (isKerasFullyCompatible) {
        val (_className, _config) = when (initializer::class) {
            GlorotUniform::class -> convertToVarianceScaling(initializer as VarianceScaling)
            GlorotNormal::class -> convertToVarianceScaling(initializer as VarianceScaling)
            HeNormal::class -> convertToVarianceScaling(initializer as VarianceScaling)
            HeUniform::class -> convertToVarianceScaling(initializer as VarianceScaling)
            LeCunNormal::class -> convertToVarianceScaling(initializer as VarianceScaling)
            LeCunUniform::class -> convertToVarianceScaling(initializer as VarianceScaling)
            else -> throw IllegalStateException("${initializer::class.simpleName} is not supported yet!")
        }

        className = _className
        config = _config
    } else {
        className = when (initializer::class) {
            GlorotUniform::class -> INITIALIZER_GLOROT_UNIFORM
            GlorotNormal::class -> INITIALIZER_GLOROT_NORMAL
            HeNormal::class -> INITIALIZER_HE_NORMAL
            HeUniform::class -> INITIALIZER_HE_UNIFORM
            LeCunNormal::class -> INITIALIZER_LECUN_NORMAL
            LeCunUniform::class -> INITIALIZER_LECUN_UNIFORM
            else -> throw IllegalStateException("${initializer::class.simpleName} is not supported yet!")
        }
        config = KerasInitializerConfig(seed = 12)
    }

    return KerasInitializer(class_name = className, config = config)
}

private fun convertToVarianceScaling(initializer: VarianceScaling): Pair<String, KerasInitializerConfig> {
    return Pair(
        INITIALIZER_VARIANCE_SCALING, KerasInitializerConfig(
            seed = initializer.seed.toInt(),
            scale = initializer.scale,
            mode = convertMode(initializer.mode),
            distribution = convertDistribution(initializer.distribution)
        )
    )
}

private fun convertDistribution(distribution: Distribution): String {
    return when (distribution) {
        Distribution.TRUNCATED_NORMAL -> "truncated_normal"
        Distribution.UNIFORM -> "uniform"
        Distribution.UNTRUNCATED_NORMAL -> "untruncated_normal"
    }
}

private fun convertMode(mode: Mode): String {
    return when (mode) {
        Mode.FAN_IN -> "fan_in"
        Mode.FAN_OUT -> "fan_out"
        Mode.FAN_AVG -> "fan_avg"
    }
}

private fun convertPadding(padding: ConvPadding): KerasPadding {
    return when (padding) {
        ConvPadding.SAME -> KerasPadding.Same
        ConvPadding.VALID -> KerasPadding.Valid
        ConvPadding.FULL -> KerasPadding.Full
    }
}

private fun convertToKerasActivation(activation: Activations): String? {
    return when (activation) {
        Activations.Relu -> ACTIVATION_RELU
        Activations.Sigmoid -> ACTIVATION_SIGMOID
        Activations.Softmax -> ACTIVATION_SOFTMAX
        Activations.Linear -> ACTIVATION_LINEAR
        Activations.Tanh -> ACTIVATION_TANH
        Activations.Relu6 -> ACTIVATION_RELU6
        Activations.Elu -> ACTIVATION_ELU
        Activations.Selu -> ACTIVATION_SELU
        Activations.LogSoftmax -> ACTIVATION_LOG_SOFTMAX
        Activations.Exponential -> ACTIVATION_EXP
        Activations.SoftPlus -> ACTIVATION_SOFTPLUS
        Activations.SoftSign -> ACTIVATION_SOFTSIGN
        Activations.HardSigmoid -> ACTIVATION_HARD_SIGMOID
        Activations.Swish -> ACTIVATION_SWISH
    }
}

private fun createKerasMaxPooling2D(layer: MaxPool2D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        padding = convertPadding(layer.padding),
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_MAX_POOLING_2D, config = configX)
}

private fun createKerasAvgPooling2D(layer: AvgPool2D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        padding = convertPadding(layer.padding),
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_AVG_POOLING_2D, config = configX)
}

private fun createKerasFlatten(layer: Flatten): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_FLATTEN, config = configX)
}

private fun createKerasConv2D(layer: Conv2D, isKerasFullyCompatible: Boolean): KerasLayer {
    val kernelSize = layer.kernelSize.map { it.toInt() }.toList()
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = kernelSize,
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer, isKerasFullyCompatible),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer, isKerasFullyCompatible),
        padding = convertPadding(layer.padding),
        name = layer.name,
        use_bias = true
    )
    return KerasLayer(class_name = LAYER_CONV2D, config = configX)
}

private fun createKerasZeroPadding2D(layer: ZeroPadding2D): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        padding = KerasPadding.ZeroPadding2D(layer.padding)
    )
    return KerasLayer(class_name = LAYER_ZERO_PADDING_2D, config = configX)
}