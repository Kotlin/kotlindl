package api.inference.keras

import api.core.Sequential
import api.core.activation.Activations
import api.core.initializer.*
import api.core.layer.Dense
import api.core.layer.Flatten
import api.core.layer.Input
import api.core.layer.Layer
import api.core.layer.twodim.AvgPool2D
import api.core.layer.twodim.Conv2D
import api.core.layer.twodim.ConvPadding
import api.core.layer.twodim.MaxPool2D
import api.inference.keras.config.KerasInitializer
import api.inference.keras.config.KerasLayer
import api.inference.keras.config.KerasSequentialModel
import api.inference.keras.config.LayerConfig
import com.beust.klaxon.Klaxon
import java.io.File

/**
 * Loads a Keras model from json file with model configuration.
 *
 * @param [jsonConfigFile] File containing model configuration.
 * @return Non-compiled and non-trained Sequential model.
 */
fun loadKerasModel(
    jsonConfigFile: File
): Sequential {
    val pair = loadKerasLayers(jsonConfigFile)
    val layers = pair.first
    val input: Input = pair.second

    return Sequential.of(input, layers.toList())
}

/**
 * Loads a Keras model layers from json file with model configuration.
 *
 * NOTE: This method is useful in transfer learning, when you need to manipulate on layers before building the Sequential model.
 *
 * @param jsonConfigFile File containing model configuration.
 * @return Pair of <list of layers; input layer>.
 */
fun loadKerasLayers(jsonConfigFile: File): Pair<MutableList<Layer>, Input> {
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)

    val sequentialConfig = Klaxon()
        .parse<KerasSequentialModel>(jsonString)

    val layers = mutableListOf<Layer>()

    sequentialConfig!!.config!!.layers!!.forEach {
        run {
            if (!it.class_name.equals("InputLayer")) {
                val layer = convertToLayer(it)
                layers.add(layer)
            }
        }
    }

    val input: Input

    val batchInputShape = sequentialConfig.config!!.layers!!.first().config!!.batch_input_shape

    input = Input(
        batchInputShape!![1]?.toLong()!!,
        batchInputShape[2]?.toLong()!!,
        batchInputShape[3]?.toLong()!!
    )
    return Pair(layers, input)
}

private fun convertToLayer(kerasLayer: KerasLayer): Layer {
    return when (kerasLayer.class_name) {
        LAYER_CONV2D -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_FLATTEN -> createFlatten(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOLING_2D -> createMaxPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_AVG_POOLING_2D -> createAvgPooling2D(
            kerasLayer.config!!,
            kerasLayer.config.name!!
        )
        LAYER_DENSE -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        else -> throw IllegalStateException("${kerasLayer.config!!.name!!} is not supported yet!")
    }
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
        INITIALIZER_ZEROS -> GlorotNormal(seed = seed) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_ONES -> GlorotNormal(seed = seed) // instead of real initializers, because it doesn't influence on nothing
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
        INITIALIZER_CONSTANT -> Constant(initializer.config.value!!.toFloat())
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
        LINEAR -> Activations.Linear
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

private fun convertPadding(padding: String): ConvPadding {
    return when (padding) {
        "same" -> ConvPadding.SAME
        "valid" -> ConvPadding.VALID
        "full" -> ConvPadding.FULL
        else -> throw UnsupportedOperationException("The $padding is not supported!")
    }
}

private fun createFlatten(config: LayerConfig, name: String): Flatten {
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
        padding = ConvPadding.SAME,
        name = name
    )
}