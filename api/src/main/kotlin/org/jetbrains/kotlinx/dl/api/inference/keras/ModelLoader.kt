/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.ELU
import org.jetbrains.kotlinx.dl.api.core.layer.activation.LeakyReLU
import org.jetbrains.kotlinx.dl.api.core.layer.activation.ReLU
import org.jetbrains.kotlinx.dl.api.core.layer.activation.ThresholdedReLU
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.DepthwiseConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.SeparableConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.*
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.GlobalAvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Cropping2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Reshape
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.ZeroPadding2D
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
    val pair = loadSequentialModelLayers(configuration)
    val input: Input = pair.first
    val layers = pair.second

    return Sequential.of(input, *layers.toList().toTypedArray())
}

/**
 * Loads a [Sequential] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param jsonConfigFile File containing model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadSequentialModelLayers(jsonConfigFile: File): Pair<Input, MutableList<Layer>> {
    val sequentialConfig = try {
        val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
        Klaxon()
            .converter(PaddingConverter())
            .parse<KerasModel>(jsonString)
    } catch (e: Exception) {
        e.printStackTrace()
        try {
            Klaxon()
                .converter(PaddingConverter())
                .parse<KerasModel>(jsonConfigFile)
        } catch (e: Exception) {
            e.printStackTrace()
            throw IllegalArgumentException("JSON file: ${jsonConfigFile.name} contains invalid JSON. The model configuration could not be loaded from this file.")
        }
    }

    val layers = mutableListOf<Layer>()

    (sequentialConfig as KerasModel).config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToSequentialLayer(it)
                layers.add(layer)
            }
        }
    }

    val input: Input

    val firstLayer = sequentialConfig.config!!.layers!!.first()
    val inputLayerName =
        if (firstLayer.class_name.equals("InputLayer")) firstLayer.config!!.name ?: "input" else "input"
    val batchInputShape = sequentialConfig.config.layers!!.first().config!!.batch_input_shape

    // TODO: write more universal code here
    when (batchInputShape!!.size) {
        3 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                name = inputLayerName
            )
        }
        4 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
        else -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
    }

    return Pair(input, layers)
}

private fun convertToSequentialLayer(
    kerasLayer: KerasLayer
): Layer {
    return when (kerasLayer.class_name) {
        LAYER_CONV2D -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DEPTHWISE_CONV2D -> createDepthwiseConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_SEPARABLE_CONV2D -> createSeparableConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_FLATTEN -> createFlatten(kerasLayer.config!!.name!!)
        LAYER_RESHAPE -> createReshape(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOLING_2D -> createMaxPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVG_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVERAGE_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_DENSE -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ZERO_PADDING_2D -> createZeroPadding2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_CROPPING_2D -> createCropping2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_BATCH_NORM -> createBatchNorm(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ACTIVATION -> createActivationLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_RELU -> createReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ELU -> createELULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_LEAKY_RELU -> createLeakyReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_THRESHOLDED_RELU -> createThresholdedReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DROPOUT -> createDropoutLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_GLOBAL_AVG_POOLING_2D -> createGlobalAvgPooling2D(
            kerasLayer.config!!.name!!
        )
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported for Sequential model!")
    }
}


/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadFunctionalModelConfiguration(
    configuration: File
): Functional {
    return Functional.of(loadFunctionalModelLayers(configuration).toList())
}

/**
 * Loads a [Functional] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Functional model.
 *
 * @param jsonConfigFile File containing model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadFunctionalModelLayers(jsonConfigFile: File): MutableList<Layer> {
    val functionalConfig = try {
        val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
        Klaxon()
            .converter(PaddingConverter())
            .parse<KerasModel>(jsonString)
    } catch (e: Exception) {
        e.printStackTrace()
        try {
            Klaxon()
                .converter(PaddingConverter())
                .parse<KerasModel>(jsonConfigFile)
        } catch (e: Exception) {
            e.printStackTrace()
            throw IllegalArgumentException("JSON file: ${jsonConfigFile.name} contains invalid JSON. The model configuration could not be loaded from this file.")
        }
    }

    val layers = mutableListOf<Layer>()
    val layersByNames = mutableMapOf<String, Layer>()

    val input: Input

    val firstLayer = (functionalConfig as KerasModel).config!!.layers!!.first()
    val batchInputShape =
        firstLayer.config!!.batch_input_shape
    val inputLayerName =
        if (firstLayer.class_name.equals("InputLayer")) firstLayer.config.name ?: "input" else "input"

    // TODO: write more universal code here
    val size = batchInputShape!!.size
    when (size) {
        3 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                name = inputLayerName
            )
        }
        4 -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
        else -> {
            input = Input(
                batchInputShape[1]?.toLong()!!,
                batchInputShape[2]?.toLong()!!,
                batchInputShape[3]?.toLong()!!,
                name = inputLayerName
            )
        }
    }

    layers.add(input)
    layersByNames[input.name] = input

    functionalConfig.config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it, layersByNames)
                layers.add(layer)
                layersByNames[layer.name] = layer
            }
        }
    }

    return layers
}

private fun convertToLayer(
    kerasLayer: KerasLayer,
    layersByName: MutableMap<String, Layer>
): Layer {
    val layer = when (kerasLayer.class_name) {
        LAYER_CONV2D -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DEPTHWISE_CONV2D -> createDepthwiseConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_SEPARABLE_CONV2D -> createSeparableConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_FLATTEN -> createFlatten(kerasLayer.config!!.name!!)
        LAYER_RESHAPE -> createReshape(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOLING_2D -> createMaxPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVG_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVERAGE_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_DENSE -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ZERO_PADDING_2D -> createZeroPadding2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_CROPPING_2D -> createCropping2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_BATCH_NORM -> createBatchNorm(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ACTIVATION -> createActivationLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_RELU -> createReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ELU -> createELULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_LEAKY_RELU -> createLeakyReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_THRESHOLDED_RELU -> createThresholdedReLULayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_DROPOUT -> createDropoutLayer(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_ADD -> createAddLayer(kerasLayer.config!!.name!!)
        LAYER_AVERAGE -> createAverageLayer(kerasLayer.config!!.name!!)
        LAYER_SUBTRACT -> createSubtractLayer(
            kerasLayer.config!!.name!!
        )
        LAYER_MAXIMUM -> createMaximumLayer(kerasLayer.config!!.name!!)
        LAYER_MINIMUM -> createMinimumLayer(kerasLayer.config!!.name!!)
        LAYER_MULTIPLY -> createMultiplyLayer(
            kerasLayer.config!!.name!!
        )
        LAYER_CONCATENATE -> createConcatenateLayer(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_GLOBAL_AVG_POOLING_2D -> createGlobalAvgPooling2D(
            kerasLayer.config!!.name!!
        )
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported yet!")
    }

    val inboundLayers = mutableListOf<Layer>()
    if (kerasLayer.class_name != LAYER_INPUT) {
        val inboundNodes = kerasLayer.inbound_nodes!! as List<List<List<Any>>>
        inboundNodes[0].forEach { inboundNode ->
            check(inboundNode.isNotEmpty()) { "This .json config is incorrect and could not be parsed! The list of inbound nodes for layer ${layer.name} could not be empty on this level!" }
            layersByName[inboundNode[0] as String]?.let { inboundLayers.add(it) }

        }
        layer.inboundLayers = inboundLayers
    }


    return layer
}

private fun createGlobalAvgPooling2D(
    name: String
): Layer {
    return GlobalAvgPool2D(
        name = name
    )
}

private fun createAddLayer(
    name: String
): Layer {
    return Add(
        name = name
    )
}

private fun createSubtractLayer(
    name: String
): Layer {
    return Subtract(
        name = name
    )
}

private fun createAverageLayer(
    name: String
): Layer {
    return Average(
        name = name
    )
}

private fun createMaximumLayer(
    name: String
): Layer {
    return Maximum(
        name = name
    )
}

private fun createMinimumLayer(
    name: String
): Layer {
    return Minimum(
        name = name
    )
}

private fun createMultiplyLayer(
    name: String
): Layer {
    return Multiply(
        name = name
    )
}

private fun createConcatenateLayer(
    config: LayerConfig,
    name: String
): Layer {
    return Concatenate(
        axis = config.axis!! as Int,
        name = name
    )
}

private fun createDropoutLayer(config: LayerConfig, name: String): Layer {
    return Dropout(
        keepProbability = config.rate!!.toFloat(),
        name = name
    )
}

private fun createActivationLayer(config: LayerConfig, name: String): Layer {
    return ActivationLayer(
        activation = convertToActivation(config.activation!!),
        name = name
    )
}

private fun createReLULayer(config: LayerConfig, name: String): Layer {
    return ReLU(
        maxValue = config.max_value!!.toFloat(),
        negativeSlope = config.negative_slope!!.toFloat(),
        threshold = config.threshold!!.toFloat(),
        name = name
    )
}

private fun createELULayer(config: LayerConfig, name: String): Layer {
    return ELU(
        alpha = config.alpha!!.toFloat(),
        name = name
    )
}

private fun createLeakyReLULayer(config: LayerConfig, name: String): Layer {
    return LeakyReLU(
        alpha = config.alpha!!.toFloat(),
        name = name
    )
}

private fun createThresholdedReLULayer(config: LayerConfig, name: String): Layer {
    return ThresholdedReLU(
        theta = config.theta!!.toFloat(),
        name = name
    )
}

private fun createBatchNorm(config: LayerConfig, name: String): Layer {
    return BatchNorm(
        axis = config.axis!! as List<Int>,
        momentum = config.momentum!!,
        center = config.center!!,
        epsilon = config.epsilon!!,
        scale = config.scale!! as Boolean,
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

private fun createReshape(config: LayerConfig, name: String): Reshape {
    return Reshape(name = name, targetShape = config.target_shape!!)
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
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createDepthwiseConv2D(
    config: LayerConfig,
    name: String
): DepthwiseConv2D {
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

    return DepthwiseConv2D(
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createSeparableConv2D(
    config: LayerConfig,
    name: String
): SeparableConv2D {
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

    return SeparableConv2D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        dilations = addedOnesDilation,
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        pointwiseInitializer = convertToInitializer(config.pointwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
        name = name
    )
}

private fun createZeroPadding2D(
    config: LayerConfig,
    name: String
): ZeroPadding2D {
    assert(config.padding is KerasPadding.ZeroPadding2D)
    return ZeroPadding2D(
        (config.padding as KerasPadding.ZeroPadding2D).padding,
        config.data_format,
        name
    )
}

private fun createCropping2D(
    config: LayerConfig,
    name: String
): Cropping2D {
    val cropping = config.cropping!!.map { it.toIntArray() }.toTypedArray()
    return Cropping2D(
        cropping,
        name
    )
}
