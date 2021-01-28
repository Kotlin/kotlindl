/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.*
import org.jetbrains.kotlinx.dl.api.core.layer.merge.Add
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.*
import org.jetbrains.kotlinx.dl.api.inference.keras.config.*
import java.io.File

/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadModelConfiguration(
    configuration: File
): Sequential {
    val pair = loadModelLayers(configuration)
    val input: Input = pair.first
    val layers = pair.second

    return Sequential.of(input, layers.toList())
}

/**
 * Loads a [Sequential] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param jsonConfigFile File containing model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadModelLayers(jsonConfigFile: File): Pair<Input, MutableList<Layer>> {
    val sequentialConfig = try {
        val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
        Klaxon()
            .converter(PaddingConverter())
            .parse<KerasSequentialModel>(jsonString)
    } catch (e: Exception) {
        e.printStackTrace()
        try {
            Klaxon()
                .converter(PaddingConverter())
                .parse<KerasSequentialModel>(jsonConfigFile)
        } catch (e: Exception) {
            e.printStackTrace()
            throw IllegalArgumentException("JSON file: ${jsonConfigFile.name} contains invalid JSON. The model configuration could not be loaded from this file.")
        }
    }

    val layers = mutableListOf<Layer>()

    (sequentialConfig as KerasSequentialModel).config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it, layers)
                layers.add(layer)
            }
        }
    }

    layers.removeAt(174)

    val input: Input

    val batchInputShape = sequentialConfig.config!!.layers!!.first().config!!.batch_input_shape

    // TODO: write more universal code here
    val size = batchInputShape!!.size
    if (size == 3) {
        input = Input(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!
        )
    } else if (size == 4) {
        input = Input(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!,
            batchInputShape[3]?.toLong()!!
        )
    } else {
        input = Input(
            batchInputShape!![1]?.toLong()!!,
            batchInputShape[2]?.toLong()!!,
            batchInputShape[3]?.toLong()!!
        )
    }

    return Pair(input, layers)
}

private fun convertToLayer(kerasLayer: KerasLayer, layers: MutableList<Layer>): Layer {
    return when (kerasLayer.class_name) {
        LAYER_CONV2D -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_FLATTEN -> createFlatten(kerasLayer.config!!.name!!)
        LAYER_MAX_POOLING_2D -> createMaxPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVG_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_DENSE -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ZERO_PADDING_2D -> createZeroPadding2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_BATCH_NORM -> createBatchNorm(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ACTIVATION -> createActivationLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_LSTM -> createLstmLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DROPOUT -> createDropoutLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ADD -> createAddLayer(kerasLayer.inbound_nodes, kerasLayer.config!!.name!!, layers)
        LAYER_GLOBAL_AVG_POOLING_2D -> createGlobalAvgPooling2D(kerasLayer.config!!, kerasLayer.config.name!!)
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported yet!")
    }
}

private fun createGlobalAvgPooling2D(config: LayerConfig, name: String): Layer {
    return Dropout(
        name = name
    )// TODO: write correct filling
}

private fun createAddLayer(inboundNodes: List<List<List<Any>>>?, name: String, layers: MutableList<Layer>): Layer {

    val mergedLayers = mutableListOf<Layer>()

    inboundNodes!![0].forEach { inboundNode ->
        layers.find { it.name == inboundNode[0] as String }?.let { mergedLayers.add(it) }
    }

    return Add(
        mergedLayers = mergedLayers,
        name = name
    )
}

private fun createDropoutLayer(config: LayerConfig, name: String): Layer {
    return Dropout(
        keepProbability = config.rate!!.toFloat(),
        name = name
    )
}

private fun createLstmLayer(config: LayerConfig, name: String): Layer {
    return LSTM(
        units = config.units!!,
        activation = convertToActivation(config.activation!!),
        recurrentActivation = convertToActivation(config.recurrent_activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        useBias = config.use_bias!!,
        unitForgetBias = config.unit_forget_bias!!,
        dropout = config.dropout!!.toFloat(),
        recurrentDropout = config.recurrent_dropout!!.toFloat(),
        returnSequences = config.return_sequences!!,
        returnState = config.return_state!!,
        goBackwards = config.go_backwards!!,
        stateful = config.stateful!!,
        timeMajor = config.time_major!!,
        unroll = config.unroll!!,
        name = name
    )
}

private fun createActivationLayer(config: LayerConfig, name: String): Layer {
    return ActivationLayer(
        activation = convertToActivation(config.activation!!),
        name = name
    )
}

private fun createBatchNorm(config: LayerConfig, name: String): Layer {
    return BatchNorm(
        axis = config.axis!!,
        momentum = config.momentum!!,
        center = config.center!!,
        epsilon = config.epsilon!!,
        scale = config.scale!!,
        gammaInitializer = convertToInitializer(config.gamma_initializer!!),
        betaInitializer = convertToInitializer(config.beta_initializer!!),
        movingMeanInitializer = convertToInitializer(config.moving_mean_initializer!!),
        movingVarianceInitializer = convertToInitializer(config.moving_variance_initializer!!),
        name = name
    )
}

private fun createDense(config: LayerConfig, name: String): Dense {
    return Dense(
        outputSize = config.units!!,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        name = name
    )
}

private fun convertToInitializer(initializer: KerasInitializer): Initializer {
    val seed = if (initializer.config!!.seed != null) {
        initializer.config.seed!!.toLong()
    } else 12L

    return when (initializer.class_name!!) {
        INITIALIZER_GLOROT_UNIFORM -> GlorotUniform(seed = seed)
        INITIALIZER_GLOROT_NORMAL -> GlorotNormal(seed = seed)
        INITIALIZER_HE_NORMAL -> HeNormal(seed = seed)
        INITIALIZER_HE_UNIFORM -> HeUniform(seed = seed)
        INITIALIZER_LECUN_NORMAL -> LeCunNormal(seed = seed)
        INITIALIZER_LECUN_UNIFORM -> LeCunUniform(seed = seed)
        INITIALIZER_ZEROS -> RandomUniform(
            seed = seed,
            minVal = 0.0f,
            maxVal = 0.0f
        ) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_CONSTANT -> RandomUniform(
            seed = seed,
            minVal = 0.0f,
            maxVal = 0.0f
        ) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_ONES -> RandomUniform(
            seed = seed,
            minVal = 1.0f,
            maxVal = 1.0f
        ) // instead of real initializers, because it doesn't influence on nothing*/
        INITIALIZER_RANDOM_NORMAL -> RandomNormal(
            seed = seed,
            mean = initializer.config.mean!!.toFloat(),
            stdev = initializer.config.stddev!!.toFloat()
        )
        INITIALIZER_RANDOM_UNIFORM -> RandomUniform(
            seed = seed,
            minVal = initializer.config.minval!!.toFloat(),
            maxVal = initializer.config.maxval!!.toFloat()
        )
        INITIALIZER_TRUNCATED_NORMAL -> TruncatedNormal(seed = seed)
        INITIALIZER_VARIANCE_SCALING -> convertVarianceScaling(initializer)
        /*INITIALIZER_CONSTANT -> Constant(initializer.config.value!!.toFloat())*/
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
    }
}

private fun convertVarianceScaling(initializer: KerasInitializer): Initializer {
    val seed = if (initializer.config!!.seed != null) {
        initializer.config.seed!!.toLong()
    } else 12L

    val config = initializer.config
    val scale = config.scale!!
    val mode: Mode = convertMode(config.mode!!)
    val distribution: Distribution = convertDistribution(config.distribution!!)
    return if (scale == 2.0 && mode == Mode.FAN_IN) {
        when (distribution) {
            Distribution.UNIFORM -> HeUniform(seed)
            Distribution.TRUNCATED_NORMAL -> {
                HeNormal(seed)
            }
            else -> VarianceScaling(scale, mode, distribution, seed)
        }
    } else {
        when (mode) {
            Mode.FAN_IN -> {
                when (distribution) {
                    Distribution.UNIFORM -> LeCunUniform(seed)
                    Distribution.TRUNCATED_NORMAL -> {
                        LeCunNormal(seed)
                    }
                    else -> VarianceScaling(scale, mode, distribution, seed)
                }
            }
            Mode.FAN_AVG -> {
                when (distribution) {
                    Distribution.UNIFORM -> GlorotUniform(seed)
                    Distribution.TRUNCATED_NORMAL -> {
                        GlorotNormal(seed)
                    }
                    else -> VarianceScaling(scale, mode, distribution, seed)
                }
            }
            else -> VarianceScaling(scale, mode, distribution, seed)
        }
    }
}

private fun convertDistribution(distribution: String): Distribution {
    return when (distribution) {
        "truncated_normal" -> Distribution.TRUNCATED_NORMAL
        "uniform" -> Distribution.UNIFORM
        "untruncated_normal" -> Distribution.UNTRUNCATED_NORMAL
        else -> Distribution.TRUNCATED_NORMAL
    }
}

private fun convertMode(mode: String): Mode {
    return when (mode) {
        "fan_in" -> Mode.FAN_IN
        "fan_out" -> Mode.FAN_OUT
        "fan_avg" -> Mode.FAN_AVG
        else -> Mode.FAN_AVG
    }
}

private fun convertToActivation(activation: String): Activations {
    return when (activation) {
        ACTIVATION_RELU -> Activations.Relu
        ACTIVATION_SIGMOID -> Activations.Sigmoid
        ACTIVATION_SOFTMAX -> Activations.Softmax
        ACTIVATION_LINEAR -> Activations.Linear
        ACTIVATION_TANH -> Activations.Tanh
        ACTIVATION_RELU6 -> Activations.Relu6
        ACTIVATION_ELU -> Activations.Elu
        ACTIVATION_SELU -> Activations.Selu
        ACTIVATION_LOG_SOFTMAX -> Activations.LogSoftmax
        ACTIVATION_EXP -> Activations.Exponential
        ACTIVATION_SOFTPLUS -> Activations.SoftPlus
        ACTIVATION_SOFTSIGN -> Activations.SoftSign
        ACTIVATION_HARD_SIGMOID -> Activations.HardSigmoid
        ACTIVATION_SWISH -> Activations.Swish
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

private fun createMaxPooling2D(config: LayerConfig, name: String): MaxPool2D {
    val poolSize = config.pool_size!!.toIntArray()
    val addedOnesPoolSize = IntArray(4)
    addedOnesPoolSize[0] = 1
    addedOnesPoolSize[1] = poolSize[0]
    addedOnesPoolSize[2] = poolSize[1]
    addedOnesPoolSize[3] = 1

    val strides = config.strides!!.toIntArray()
    val addedOnesStrides = IntArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    return MaxPool2D(addedOnesPoolSize, addedOnesStrides, padding = convertPadding(config.padding!!), name = name)
}

private fun createAvgPooling2D(config: LayerConfig, name: String): AvgPool2D {
    val poolSize = config.pool_size!!.toIntArray()
    val addedOnesPoolSize = IntArray(4)
    addedOnesPoolSize[0] = 1
    addedOnesPoolSize[1] = poolSize[0]
    addedOnesPoolSize[2] = poolSize[1]
    addedOnesPoolSize[3] = 1

    val strides = config.strides!!.toIntArray()
    val addedOnesStrides = IntArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    return AvgPool2D(addedOnesPoolSize, addedOnesStrides, padding = convertPadding(config.padding!!), name = name)
}

private fun convertPadding(padding: KerasPadding): ConvPadding {
    return when (padding) {
        is KerasPadding.Same -> ConvPadding.SAME
        is KerasPadding.Valid -> ConvPadding.VALID
        is KerasPadding.Full -> ConvPadding.FULL
        else -> throw UnsupportedOperationException("The $padding is not supported!")
    }
}

private fun createFlatten(name: String): Flatten {
    return Flatten(name = name)
}

private fun createConv2D(config: LayerConfig, name: String): Conv2D {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    val dilation = config.dilation_rate!!.map { it.toLong() }.toLongArray()
    val addedOnesDilation = LongArray(4)
    addedOnesDilation[0] = 1
    addedOnesDilation[1] = dilation[0]
    addedOnesDilation[2] = dilation[1]
    addedOnesDilation[3] = 1

    return Conv2D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        padding = convertPadding(config.padding!!),
        name = name
    )
}

private fun createZeroPadding2D(config: LayerConfig, name: String) : ZeroPadding2D {
    assert(config.padding is KerasPadding.ZeroPadding2D)
    return ZeroPadding2D(
        (config.padding as KerasPadding.ZeroPadding2D).padding,
        config.data_format,
        name
    )
}
