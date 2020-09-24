package api.inference.keras

import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.*
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Layer
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.MaxPool2D
import api.inference.keras.config.*
import com.beust.klaxon.Klaxon
import java.io.File

/**
 * Saves model description as json configuration file fully compatible with the Keras TensorFlow framework.
 *
 * @param jsonConfigFile File to write model configuration.
 */
fun Sequential.saveConfig(jsonConfigFile: File) {
    val kerasLayers = mutableListOf<KerasLayer>()
    this.layers.forEach {
        run {
            val layer = convertToKerasLayer(it)
            kerasLayers.add(layer)
        }
    }

    val inputShape = this.firstLayer.packedDims.map { it.toInt() }

    (kerasLayers.first().config as LayerConfig).batch_input_shape =
        listOf(null, inputShape[0], inputShape[1], inputShape[2])

    val config = SequentialConfig(name = "", layers = kerasLayers)
    val sequentialConfig = KerasSequentialModel(config = config)

    val jsonString2 = Klaxon().toJsonString(sequentialConfig)

    jsonConfigFile.writeText(jsonString2, Charsets.UTF_8)
}

private fun convertToKerasLayer(layer: Layer): KerasLayer {
    return when (layer::class) {
        Conv2D::class -> createKerasConv2D(layer as Conv2D)
        Flatten::class -> createKerasFlatten(layer as Flatten)
        MaxPool2D::class -> createKerasMaxPooling2D(layer as MaxPool2D)
        Dense::class -> createKerasDense(layer as Dense)
        else -> throw IllegalStateException("${layer.name} is not supported yet!")
    }
}

private fun createKerasDense(layer: Dense): KerasLayer {
    val configX = LayerConfig(
        dtype = DATATYPE_FLOAT32,
        units = layer.outputSize,
        name = layer.name,
        use_bias = true,
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer)
    )
    return KerasLayer(class_name = LAYER_DENSE, config = configX)
}

private fun convertToKerasInitializer(initializer: Initializer): KerasInitializer? {
    val className = when (initializer::class) {
        GlorotUniform::class -> INITIALIZER_GLOROT_UNIFORM
        GlorotNormal::class -> INITIALIZER_GLOROT_NORMAL
        Zeros::class -> INITIALIZER_ZEROS
        Ones::class -> INITIALIZER_ONES
        else -> throw IllegalStateException("${initializer::class.simpleName} is not supported yet!")
    }
    val config = KerasInitializerConfig(seed = 12)
    return KerasInitializer(class_name = className, config = config)
}

private fun convertToKerasActivation(activation: Activations): String? {
    return when (activation) {
        Activations.Relu -> ACTIVATION_RELU
        Activations.Sigmoid -> ACTIVATION_SIGMOID
        Activations.Softmax -> ACTIVATION_SOFTMAX
        Activations.Linear -> LINEAR
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

private fun createKerasMaxPooling2D(layer: MaxPool2D): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name,
        padding = PADDING_SAME,
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = LAYER_MAX_POOLING_2D, config = configX)
}

private fun createKerasFlatten(layer: Flatten): KerasLayer {
    val configX = LayerConfig(
        data_format = CHANNELS_LAST,
        dtype = DATATYPE_FLOAT32,
        name = layer.name
    )
    return KerasLayer(class_name = LAYER_FLATTEN, config = configX)
}

private fun createKerasConv2D(layer: Conv2D): KerasLayer {
    val kernelSize = layer.kernelSize.map { it.toInt() }.toList()
    val configX = LayerConfig(
        filters = layer.filters.toInt(),
        kernel_size = kernelSize,
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        dilation_rate = listOf(layer.dilations[1].toInt(), layer.dilations[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasInitializer(layer.kernelInitializer),
        bias_initializer = convertToKerasInitializer(layer.biasInitializer),
        padding = PADDING_SAME,
        name = layer.name,
        use_bias = true
    )
    return KerasLayer(class_name = LAYER_CONV2D, config = configX)
}
