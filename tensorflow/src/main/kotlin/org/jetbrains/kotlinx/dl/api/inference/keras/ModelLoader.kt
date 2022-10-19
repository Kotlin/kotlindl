/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import com.beust.klaxon.Klaxon
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.*
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.TrainableLayer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.*
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.*
import org.jetbrains.kotlinx.dl.api.core.layer.core.ActivationLayer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.merge.*
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.*
import org.jetbrains.kotlinx.dl.api.core.layer.regularization.Dropout
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.*
import org.jetbrains.kotlinx.dl.api.core.regularizer.L1
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2
import org.jetbrains.kotlinx.dl.api.core.regularizer.L2L1
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.inference.keras.config.*
import java.io.File

/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadSequentialModelConfiguration(
    configuration: File,
    inputShape: IntArray? = null
): Sequential {
    val sequentialConfig = loadSerializedModel(configuration)
    return deserializeSequentialModel(sequentialConfig, inputShape)
}

internal fun deserializeSequentialModel(sequentialConfig: KerasModel?, inputShape: IntArray? = null): Sequential {
    val pair = loadSequentialModelLayers(sequentialConfig, inputShape)
    val input: Input = pair.first
    val layers = pair.second

    return Sequential.of(input, *layers.toList().toTypedArray())
}

/**
 * Loads a [Sequential] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param config Model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadSequentialModelLayers(
    config: KerasModel?,
    inputShape: IntArray? = null
): Pair<Input, List<Layer>> {
    val kerasLayers = config!!.config!!.layers!!

    val input = createInputLayer(kerasLayers.first(), inputShape)
    val layers = kerasLayers.filter { !it.class_name.equals(LAYER_INPUT) }.mapTo(mutableListOf()) {
        convertToLayer(it)
    }

    return Pair(input, layers)
}

private fun convertToLayer(
    kerasLayer: KerasLayer
): Layer {
    return when (kerasLayer.class_name) {
        // Core layers
        LAYER_ACTIVATION -> createActivationLayer(kerasLayer.config!!)
        LAYER_DENSE -> createDenseLayer(kerasLayer.config!!)
        LAYER_PERMUTE -> createPermuteLayer(kerasLayer.config!!)
        // Convolution layers
        LAYER_CONV1D -> createConv1DLayer(kerasLayer.config!!)
        LAYER_CONV2D -> createConv2DLayer(kerasLayer.config!!)
        LAYER_CONV3D -> createConv3DLayer(kerasLayer.config!!)
        LAYER_CONV1D_TRANSPOSE -> createConv1DTransposeLayer(kerasLayer.config!!)
        LAYER_CONV2D_TRANSPOSE -> createConv2DTransposeLayer(kerasLayer.config!!)
        LAYER_CONV3D_TRANSPOSE -> createConv3DTransposeLayer(kerasLayer.config!!)
        LAYER_DEPTHWISE_CONV2D -> createDepthwiseConv2DLayer(kerasLayer.config!!)
        LAYER_SEPARABLE_CONV2D -> createSeparableConv2DLayer(kerasLayer.config!!)
        // Pooling layers
        LAYER_MAX_POOL_1D -> createMaxPool1DLayer(kerasLayer.config!!)
        LAYER_MAX_POOL_2D -> createMaxPool2DLayer(kerasLayer.config!!)
        LAYER_MAX_POOL_3D -> createMaxPool3DLayer(kerasLayer.config!!)
        LAYER_AVG_POOL_1D -> createAvgPool1DLayer(kerasLayer.config!!)
        LAYER_AVG_POOL_2D -> createAvgPool2DLayer(kerasLayer.config!!)
        LAYER_AVG_POOL_3D -> createAvgPool3DLayer(kerasLayer.config!!)
        LAYER_GLOBAL_MAX_POOL_1D -> GlobalMaxPool1D()
        LAYER_GLOBAL_MAX_POOL_2D -> GlobalMaxPool2D()
        LAYER_GLOBAL_MAX_POOL_3D -> GlobalMaxPool3D()
        LAYER_GLOBAL_AVG_POOL_1D -> GlobalAvgPool1D()
        LAYER_GLOBAL_AVG_POOL_2D -> GlobalAvgPool2D()
        LAYER_GLOBAL_AVG_POOL_3D -> GlobalAvgPool3D()
        // Recurrent layers
        // Normalization layers
        LAYER_BATCH_NORM -> createBatchNormLayer(kerasLayer.config!!)
        // Regularization layers
        LAYER_DROPOUT -> createDropoutLayer(kerasLayer.config!!)
        // Attention layers
        // Reshaping layers
        LAYER_FLATTEN -> Flatten()
        LAYER_REPEAT_VECTOR -> createRepeatVectorLayer(kerasLayer.config!!)
        LAYER_RESHAPE -> createReshapeLayer(kerasLayer.config!!)
        LAYER_CROPPING_1D -> createCropping1DLayer(kerasLayer.config!!)
        LAYER_CROPPING_2D -> createCropping2DLayer(kerasLayer.config!!)
        LAYER_CROPPING_3D -> createCropping3DLayer(kerasLayer.config!!)
        LAYER_ZERO_PADDING_1D -> createZeroPadding1DLayer(kerasLayer.config!!)
        LAYER_ZERO_PADDING_2D -> createZeroPadding2DLayer(kerasLayer.config!!)
        LAYER_ZERO_PADDING_3D -> createZeroPadding3DLayer(kerasLayer.config!!)
        LAYER_UP_SAMPLING_1D -> createUpSampling1DLayer(kerasLayer.config!!)
        LAYER_UP_SAMPLING_2D -> createUpSampling2DLayer(kerasLayer.config!!)
        LAYER_UP_SAMPLING_3D -> createUpSampling3DLayer(kerasLayer.config!!)
        // Merging layers
        LAYER_ADD -> Add()
        LAYER_AVERAGE -> Average()
        LAYER_SUBTRACT -> Subtract()
        LAYER_MAXIMUM -> Maximum()
        LAYER_MINIMUM -> Minimum()
        LAYER_MULTIPLY -> Multiply()
        LAYER_CONCATENATE -> createConcatenateLayer(kerasLayer.config!!)
        LAYER_DOT -> createDotLayer(kerasLayer.config!!)
        // Locally-connected layers
        // Activation layers
        LAYER_RELU -> createReLULayer(kerasLayer.config!!)
        LAYER_ELU -> createELULayer(kerasLayer.config!!)
        LAYER_PRELU -> createPReLULayer(kerasLayer.config!!)
        LAYER_LEAKY_RELU -> createLeakyReLULayer(kerasLayer.config!!)
        LAYER_THRESHOLDED_RELU -> createThresholdedReLULayer(kerasLayer.config!!)
        LAYER_SOFTMAX -> createSoftmaxLayer(kerasLayer.config!!)
        else -> throw IllegalStateException("${kerasLayer.class_name} is not supported yet!")
    }.apply {
        if (this is TrainableLayer) {
            isTrainable = kerasLayer.config?.trainable ?: isTrainable
        }
        name = kerasLayer.config?.name ?: name
    }
}


/**
 * Loads a [Sequential] model from json file with model configuration.
 *
 * @param [configuration] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
internal fun loadFunctionalModelConfiguration(
    configuration: File,
    inputShape: IntArray? = null
): Functional {
    val functionalConfig = loadSerializedModel(configuration)
    return deserializeFunctionalModel(functionalConfig, inputShape)
}

internal fun deserializeFunctionalModel(functionalConfig: KerasModel?, inputShape: IntArray? = null) =
    Functional.of(loadFunctionalModelLayers(functionalConfig, inputShape).toList())

/**
 * Loads a [Functional] model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Functional model.
 *
 * @param config Model configuration.
 * @return Pair of <input layer; list of layers>.
 */
internal fun loadFunctionalModelLayers(config: KerasModel?, inputShape: IntArray? = null): List<Layer> {
    val layers = mutableListOf<Layer>()
    val layersByNames = mutableMapOf<String, Layer>()

    val kerasLayers = config!!.config!!.layers!!
    val input = createInputLayer(kerasLayers.first(), inputShape)
    layers.add(input)
    layersByNames[input.name] = input

    kerasLayers.forEach {
        if (!it.class_name.equals(LAYER_INPUT)) {
            val layer = convertToLayer(it, layersByNames)
            layers.add(layer)
            layersByNames[layer.name] = layer
        }
    }

    return layers
}

internal fun loadSerializedModel(jsonConfigFile: File) = try {
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

private fun convertToLayer(
    kerasLayer: KerasLayer,
    layersByName: Map<String, Layer>
): Layer {
    val layer = convertToLayer(kerasLayer)
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

private fun convertToRegularizer(regularizer: KerasRegularizer?): Regularizer? {
    return if (regularizer != null) {
        val l1 = regularizer.config!!.l1
        val l2 = regularizer.config.l2
        if (l1 != 0.0 && l2 != 0.0) {
            L2L1(l1!!.toFloat(), l2!!.toFloat())
        } else if (l1 == 0.0 && l2 != 0.0) {
            L2(l2!!.toFloat())
        } else if (l1 != 0.0 && l2 == 0.0) {
            L1(l1!!.toFloat())
        } else {
            null
        }
    } else {
        null
    }
}

private fun convertToInitializer(initializer: KerasInitializer): Initializer {
    val config = initializer.config
    val seed = config!!.seed?.toLong() ?: 12L

    return when (initializer.class_name!!) {
        INITIALIZER_GLOROT_UNIFORM -> GlorotUniform(seed = seed)
        INITIALIZER_GLOROT_NORMAL -> GlorotNormal(seed = seed)
        INITIALIZER_HE_NORMAL -> HeNormal(seed = seed)
        INITIALIZER_HE_UNIFORM -> HeUniform(seed = seed)
        INITIALIZER_LECUN_NORMAL -> LeCunNormal(seed = seed)
        INITIALIZER_LECUN_UNIFORM -> LeCunUniform(seed = seed)
        INITIALIZER_RANDOM_NORMAL -> RandomNormal(
            seed = seed,
            mean = config.mean!!.toFloat(),
            stdev = config.stddev!!.toFloat()
        )
        INITIALIZER_RANDOM_UNIFORM -> RandomUniform(
            seed = seed,
            minVal = config.minval!!.toFloat(),
            maxVal = config.maxval!!.toFloat()
        )
        INITIALIZER_VARIANCE_SCALING -> convertVarianceScalingInitializer(initializer)
        INITIALIZER_TRUNCATED_NORMAL -> TruncatedNormal(seed = seed)
        INITIALIZER_PARAMETRIZED_TRUNCATED_NORMAL -> ParametrizedTruncatedNormal(
            mean = config.mean!!.toFloat(),
            stdev = config.stddev!!.toFloat(),
            p1 = config.p1!!.toFloat(),
            p2 = config.p2!!.toFloat(),
            seed = seed
        )
        INITIALIZER_ORTHOGONAL -> Orthogonal(seed = seed, gain = config.gain!!.toFloat())
        INITIALIZER_ZEROS -> Zeros()
        INITIALIZER_ONES -> Ones()
        INITIALIZER_CONSTANT -> Constant(config.value!!.toFloat())
        INITIALIZER_IDENTITY -> Identity(config.gain?.toFloat() ?: 1f)
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
    }
}

private fun convertVarianceScalingInitializer(initializer: KerasInitializer): Initializer {
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
        ACTIVATION_TANHSHRINK -> Activations.TanhShrink
        ACTIVATION_RELU6 -> Activations.Relu6
        ACTIVATION_ELU -> Activations.Elu
        ACTIVATION_SELU -> Activations.Selu
        ACTIVATION_LOG_SOFTMAX -> Activations.LogSoftmax
        ACTIVATION_EXP -> Activations.Exponential
        ACTIVATION_SOFTPLUS -> Activations.SoftPlus
        ACTIVATION_SOFTSIGN -> Activations.SoftSign
        ACTIVATION_HARD_SIGMOID -> Activations.HardSigmoid
        ACTIVATION_SWISH -> Activations.Swish
        ACTIVATION_MISH -> Activations.Mish
        ACTIVATION_HARDSHRINK -> Activations.HardShrink
        ACTIVATION_LISHT -> Activations.LiSHT
        ACTIVATION_SNAKE -> Activations.Snake
        ACTIVATION_GELU -> Activations.Gelu
        ACTIVATION_SPARSEMAX -> Activations.Sparsemax
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

private fun convertToInterpolationMethod(interpolation: String): InterpolationMethod {
    return when (interpolation) {
        InterpolationMethod.NEAREST.methodName -> InterpolationMethod.NEAREST
        InterpolationMethod.BILINEAR.methodName -> InterpolationMethod.BILINEAR
        InterpolationMethod.BICUBIC.methodName -> InterpolationMethod.BICUBIC
        else -> throw IllegalArgumentException("Interpolation $interpolation is not supported yet!")
    }
}

/**
 * The layer creator functions should be put below.
 */

private fun createInputLayer(layer: KerasLayer, inputShape: IntArray? = null): Input {
    val inputLayerDims = if (inputShape != null) {
        inputShape.map { it.toLong() }.toLongArray()
    } else {
        val batchInputShape = layer.config!!.batch_input_shape!!
        batchInputShape.subList(1, batchInputShape.size).map { it!!.toLong() }.toLongArray()
    }

    val inputLayerName = if (layer.class_name.equals(LAYER_INPUT)) layer.config!!.name ?: "input" else "input"
    return Input(*inputLayerDims, name = inputLayerName)
}

private fun createConcatenateLayer(config: LayerConfig): Layer {
    return Concatenate(
        axis = config.axis!! as Int
    )
}

private fun createDotLayer(config: LayerConfig): Layer {
    return Dot(
        axis = config.axis!! as IntArray,
        normalize = config.normalize ?: false
    )
}

private fun createDropoutLayer(config: LayerConfig): Layer {
    return Dropout(
        rate = config.rate!!.toFloat()
    )
}

private fun createActivationLayer(config: LayerConfig): Layer {
    return ActivationLayer(
        activation = convertToActivation(config.activation!!)
    )
}

private fun createReLULayer(config: LayerConfig): Layer {
    return ReLU(
        maxValue = config.max_value!!.toFloat(),
        negativeSlope = config.negative_slope!!.toFloat(),
        threshold = config.threshold!!.toFloat()
    )
}

private fun createELULayer(config: LayerConfig): Layer {
    return ELU(
        alpha = config.alpha!!.toFloat()
    )
}

private fun createPReLULayer(config: LayerConfig): Layer {
    return PReLU(
        alphaInitializer = convertToInitializer(config.alpha_initializer!!),
        alphaRegularizer = convertToRegularizer(config.alpha_regularizer),
        sharedAxes = config.shared_axes?.toIntArray()
    )
}

private fun createLeakyReLULayer(config: LayerConfig): Layer {
    return LeakyReLU(
        alpha = config.alpha!!.toFloat()
    )
}

private fun createThresholdedReLULayer(config: LayerConfig): Layer {
    return ThresholdedReLU(
        theta = config.theta!!.toFloat()
    )
}

private fun createSoftmaxLayer(config: LayerConfig): Layer {
    val axis = when (config.axis) {
        is Int -> listOf(config.axis)
        is List<*> -> config.axis as List<Int>
        else -> throw IllegalArgumentException("Axis must be an integer or a list of integers")
    }
    return Softmax(
        axis = axis
    )
}

private fun createBatchNormLayer(config: LayerConfig): Layer {
    return BatchNorm(
        axis = config.axis!! as List<Int>,
        momentum = config.momentum!!,
        center = config.center!!,
        epsilon = config.epsilon!!,
        scale = config.scale!! as Boolean,
        gammaInitializer = convertToInitializer(config.gamma_initializer!!),
        betaInitializer = convertToInitializer(config.beta_initializer!!),
        gammaRegularizer = convertToRegularizer(config.gamma_regularizer),
        betaRegularizer = convertToRegularizer(config.beta_regularizer),
        movingMeanInitializer = convertToInitializer(config.moving_mean_initializer!!),
        movingVarianceInitializer = convertToInitializer(config.moving_variance_initializer!!)
    )
}

private fun createDenseLayer(config: LayerConfig): Layer {
    return Dense(
        outputSize = config.units!!,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        useBias = config.use_bias ?: true
    )
}

private fun createPermuteLayer(config: LayerConfig): Layer {
    return Permute(
        dims = config.dims!!
    )
}

private fun createMaxPool1DLayer(config: LayerConfig): Layer {
    return MaxPool1D(
        poolSize = intArrayOf(1, config.pool_size!![0], 1),
        strides = intArrayOf(1, config.strides!![0], 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun createMaxPool2DLayer(config: LayerConfig): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    return MaxPool2D(
        poolSize = intArrayOf(1, *poolSize, 1),
        strides = intArrayOf(1, *strides, 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun createAvgPool1DLayer(config: LayerConfig): Layer {
    return AvgPool1D(
        poolSize = intArrayOf(1, config.pool_size!![0], 1),
        strides = intArrayOf(1, config.strides!![0], 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun createAvgPool2DLayer(config: LayerConfig): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    return AvgPool2D(
        poolSize = intArrayOf(1, *poolSize, 1),
        strides = intArrayOf(1, *strides, 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun createAvgPool3DLayer(config: LayerConfig): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    return AvgPool3D(
        poolSize = intArrayOf(1, *poolSize, 1),
        strides = intArrayOf(1, *strides, 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun createMaxPool3DLayer(config: LayerConfig): Layer {
    val poolSize = config.pool_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    return MaxPool3D(
        poolSize = intArrayOf(1, *poolSize, 1),
        strides = intArrayOf(1, *strides, 1),
        padding = convertPadding(config.padding!!)
    )
}

private fun convertPadding(padding: KerasPadding): ConvPadding {
    return when (padding) {
        is KerasPadding.Same -> ConvPadding.SAME
        is KerasPadding.Valid -> ConvPadding.VALID
        is KerasPadding.Full -> ConvPadding.FULL
        else -> throw UnsupportedOperationException("The $padding is not supported!")
    }
}

private fun createRepeatVectorLayer(config: LayerConfig): Layer {
    return RepeatVector(n = config.n!!)
}

private fun createReshapeLayer(config: LayerConfig): Layer {
    return Reshape(targetShape = config.target_shape!!)
}

private fun createConv1DLayer(config: LayerConfig): Layer {
    return Conv1D(
        filters = config.filters!!,
        kernelLength = config.kernel_size!![0],
        strides = intArrayOf(1, config.strides!![0], 1),
        dilations = intArrayOf(1, config.dilation_rate!![0], 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!
    )
}

private fun createConv2DLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()
    return Conv2D(
        filters = config.filters!!,
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!
    )
}

private fun createConv3DLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()
    return Conv3D(
        filters = config.filters!!,
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!
    )
}

private fun createConv1DTransposeLayer(config: LayerConfig): Layer {
    return Conv1DTranspose(
        filters = config.filters!!,
        kernelLength = config.kernel_size!![0],
        strides = intArrayOf(1, config.strides!![0], 1),
        dilations = intArrayOf(1, config.dilation_rate!![0], 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        outputPadding = config.output_padding?.convertToOutputPadding(),
        useBias = config.use_bias!!,
    )
}

private fun createConv2DTransposeLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()
    return Conv2DTranspose(
        filters = config.filters!!,
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        outputPadding = config.output_padding?.convertToOutputPadding(),
        useBias = config.use_bias!!,
    )
}

private fun createConv3DTransposeLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()
    return Conv3DTranspose(
        filters = config.filters!!,
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToInitializer(config.kernel_initializer!!),
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        kernelRegularizer = convertToRegularizer(config.kernel_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!,
    )
}

private fun List<Int>.convertToOutputPadding(): IntArray {
    return intArrayOf(
        0, 0,
        *flatMap { padding -> listOf(padding / 2, padding - padding / 2) }.toIntArray(),
        0, 0
    )
}

private fun createDepthwiseConv2DLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()
    return DepthwiseConv2D(
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        depthwiseRegularizer = convertToRegularizer(config.depthwise_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!
    )
}

private fun createSeparableConv2DLayer(config: LayerConfig): Layer {
    val kernelSize = config.kernel_size!!.toIntArray()
    val strides = config.strides!!.toIntArray()
    val dilation = config.dilation_rate!!.toIntArray()

    return SeparableConv2D(
        filters = config.filters!!,
        kernelSize = kernelSize,
        strides = intArrayOf(1, *strides, 1),
        dilations = intArrayOf(1, *dilation, 1),
        activation = convertToActivation(config.activation!!),
        depthwiseInitializer = convertToInitializer(config.depthwise_initializer!!),
        pointwiseInitializer = convertToInitializer(config.pointwise_initializer!!),
        depthMultiplier = config.depth_multiplier!!,
        biasInitializer = convertToInitializer(config.bias_initializer!!),
        depthwiseRegularizer = convertToRegularizer(config.depthwise_regularizer),
        pointwiseRegularizer = convertToRegularizer(config.pointwise_regularizer),
        activityRegularizer = convertToRegularizer(config.activity_regularizer),
        biasRegularizer = convertToRegularizer(config.bias_regularizer),
        padding = convertPadding(config.padding!!),
        useBias = config.use_bias!!
    )
}

private fun createZeroPadding1DLayer(config: LayerConfig): Layer {
    assert(config.padding is KerasPadding.ZeroPadding1D)
    return ZeroPadding1D(
        padding = (config.padding as KerasPadding.ZeroPadding1D).padding
    )
}

private fun createZeroPadding2DLayer(config: LayerConfig): Layer {
    assert(config.padding is KerasPadding.ZeroPadding2D)
    return ZeroPadding2D(
        padding = (config.padding as KerasPadding.ZeroPadding2D).padding,
        dataFormat = config.data_format
    )
}

private fun createZeroPadding3DLayer(config: LayerConfig): Layer {
    assert(config.padding is KerasPadding.ZeroPadding3D)
    return ZeroPadding3D(
        padding = (config.padding as KerasPadding.ZeroPadding3D).padding
    )
}

private fun createCropping1DLayer(config: LayerConfig): Layer {
    val cropping = config.cropping!!.map { it as Int }.toTypedArray().toIntArray()
    return Cropping1D(
        cropping = cropping
    )
}

private fun createCropping2DLayer(config: LayerConfig): Layer {
    val cropping = config.cropping!!.map { (it as List<Int>).toIntArray() }.toTypedArray()
    return Cropping2D(
        cropping = cropping
    )
}

private fun createCropping3DLayer(config: LayerConfig): Layer {
    val cropping = config.cropping!!.map { (it as List<Int>).toIntArray() }.toTypedArray()
    return Cropping3D(
        cropping = cropping
    )
}

private fun createUpSampling1DLayer(config: LayerConfig): Layer {
    return UpSampling1D(
        size = config.size!! as Int
    )
}

private fun createUpSampling2DLayer(config: LayerConfig): Layer {
    return UpSampling2D(
        size = (config.size!! as List<Int>).toIntArray(),
        interpolation = convertToInterpolationMethod(config.interpolation!!)
    )
}

private fun createUpSampling3DLayer(config: LayerConfig): Layer {
    return UpSampling3D(
        size = (config.size!! as List<Int>).toIntArray()
    )
}
