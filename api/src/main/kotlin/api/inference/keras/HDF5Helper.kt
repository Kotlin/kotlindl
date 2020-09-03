package api.inference.keras

import api.*
import api.inference.keras.config.*
import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.initializers.*
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import com.beust.klaxon.Klaxon
import io.jhdf.HdfFile
import io.jhdf.api.Group
import java.io.File

// Keras layers
private const val LAYER_CONV2D = "Conv2D"
private const val LAYER_DENSE = "Dense"
private const val LAYER_MAX_POOLING_2D = "MaxPooling2D"
private const val LAYER_FLATTEN = "Flatten"

// Keras data types
private const val DATATYPE_FLOAT32 = "float32"

// Keras Initializers
private const val INITIALIZER_GLOROT_UNIFORM = "GlorotUniform"
private const val INITIALIZER_GLOROT_NORMAL = "GlorotNormal"
private const val INITIALIZER_ZEROS = "Zeros"
private const val INITIALIZER_ONES = "Ones"
private const val INITIALIZER_RANDOM_NORMAL = "RandomNormal"

// Keras activations
private const val ACTIVATION_RELU = "relu"
private const val ACTIVATION_SIGMOID = "sigmoid"
private const val ACTIVATION_SOFTMAX = "softmax"
private const val LINEAR = "linear"

// Layer settings
private const val CHANNELS_LAST = "channels_last"
private const val PADDING_SAME = "same"

private const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
private const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"

fun Sequential.loadWeights(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            when (it::class) {
                Dense::class -> fillDenseVariables(it.name, hdfFile, this, kernelDataPathTemplate, biasDataPathTemplate)
                Conv2D::class -> fillConv2DVariables(
                    it.name,
                    hdfFile,
                    this,
                    kernelDataPathTemplate,
                    biasDataPathTemplate
                )
                else -> println("No weights loading for ${it.name}")
            }
            this.logger.info { " Weights loaded for ${it.name}. ${it.getParams()} parameters are loaded. " }
        }
    }
    this.logger.info { "Weights are loaded." }
}

/**
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 */
fun Sequential.loadWeights(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (layerList.contains(it)) {
                when (it::class) {
                    Dense::class -> fillDenseVariables(
                        it.name,
                        hdfFile,
                        this,
                        kernelDataPathTemplate,
                        biasDataPathTemplate
                    )
                    Conv2D::class -> fillConv2DVariables(
                        it.name,
                        hdfFile,
                        this,
                        kernelDataPathTemplate,
                        biasDataPathTemplate
                    )
                    else -> println("No weights loading for ${it.name}")
                }
                this.logger.info { " Weights loaded for ${it.name}. ${it.getParams()} parameters are loaded. " }
            } else {
                when (it::class) {
                    Dense::class -> initDenseVariablesByDefaultInitializer(it.name, this)
                    Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, this)
                    else -> println("No default initialization handled for ${it.name}")
                }
                this.logger.info { " Weights initialized for ${it.name}. ${it.getParams()} parameters are initialized. " }
            }
        }
    }
    this.logger.info { "Weights are loaded." }
}

/**
 * Weights for other layers will be initialized by initializer later.
 */
fun Sequential.loadWeightsForFrozenLayers(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (!it.isTrainable) {
                when (it::class) {
                    Dense::class -> fillDenseVariables(
                        it.name,
                        hdfFile,
                        this,
                        kernelDataPathTemplate,
                        biasDataPathTemplate
                    )
                    Conv2D::class -> fillConv2DVariables(
                        it.name,
                        hdfFile,
                        this,
                        kernelDataPathTemplate,
                        biasDataPathTemplate
                    )
                    else -> println("No weights loading for ${it.name}")
                }
                this.logger.info { " Weights loaded for ${it.name}. ${it.getParams()} parameters are loaded. " }
            } else {
                when (it::class) {
                    Dense::class -> initDenseVariablesByDefaultInitializer(it.name, this)
                    Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, this)
                    else -> println("No default initialization handled for ${it.name}")
                }
                this.logger.info { " Weights initialized for ${it.name}. ${it.getParams()} parameters are initialized. " }
            }
        }
    }
    this.logger.info { "Weights are loaded." }
}

private fun initConv2DVariablesByDefaultInitializer(name: String, model: Sequential) {
    val kernelVariableName = conv2dKernelVarName(name)
    val biasVariableName = conv2dBiasVarName(name)
    runInitOps(kernelVariableName, model)
    runInitOps(biasVariableName, model)
}


private fun initDenseVariablesByDefaultInitializer(name: String, model: Sequential) {
    val kernelVariableName = denseKernelVarName(name)
    val biasVariableName = denseBiasVarName(name)
    runInitOps(kernelVariableName, model)
    runInitOps(biasVariableName, model)
}

// TODO: move to inference class
private fun runInitOps(
    variableName: String,
    model: Sequential
) {
    val assignOpName = defaultAssignOpName(variableName)

    model.session.runner()
        .addTarget(assignOpName)
        .run()
}

fun buildModelByJSONConfig(
    jsonConfigFile: File
): Sequential {
    val pair = buildLayersByKerasJSONConfig(jsonConfigFile)
    val layers = pair.first
    val input: Input = pair.second

    return Sequential.of(input, layers.toList())
}

fun buildLayersByKerasJSONConfig(jsonConfigFile: File): Pair<MutableList<Layer>, Input> {
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)

    val sequentialConfig = Klaxon()
        .parse<KerasSequentialModel>(jsonString)

    val layers = mutableListOf<Layer>()

    sequentialConfig!!.config!!.layers!!.forEach {
        run {
            if (it.class_name.equals("InputLayer")) {

            } else {
                val layer = convertToLayer(it)
                layers.add(layer)
            }
        }
    }

    val input: Input

    val batchInputShape = sequentialConfig!!.config!!.layers!!.first().config!!.batch_input_shape

    input = Input(
        batchInputShape!![1]?.toLong()!!,
        batchInputShape!![2]?.toLong()!!,
        batchInputShape!![3]?.toLong()!!
    )
    return Pair(layers, input)
}

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

// TODO: create separate configs for different initializers
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


private fun fillConv2DVariables(
    name: String,
    hdfFile: HdfFile,
    model: Sequential,
    kernelDataPathTemplate: String,
    biasDataPathTemplate: String
) {
    val kernelData = hdfFile.getDatasetByPath(kernelDataPathTemplate.format(name, name)).data
    val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data

    val kernelVariableName = conv2dKernelVarName(name)
    val biasVariableName = conv2dBiasVarName(name)
    addInitOpsToGraph(kernelVariableName, model, kernelData)
    addInitOpsToGraph(biasVariableName, model, biasData)
}


private fun fillDenseVariables(
    name: String,
    hdfFile: HdfFile,
    model: Sequential,
    kernelDataPathTemplate: String,
    biasDataPathTemplate: String
) {
    val kernelData = hdfFile.getDatasetByPath(kernelDataPathTemplate.format(name, name)).data
    val biasData = hdfFile.getDatasetByPath(biasDataPathTemplate.format(name, name)).data

    val kernelVariableName = denseKernelVarName(name)
    val biasVariableName = denseBiasVarName(name)

    addInitOpsToGraph(kernelVariableName, model, kernelData)
    addInitOpsToGraph(biasVariableName, model, biasData)
}

private fun addInitOpsToGraph(
    variableName: String,
    model: Sequential,
    kernelData: Any
) {
    val initializerName = defaultInitializerOpName(variableName)
    val assignOpName = defaultAssignOpName(variableName)

    model.populateVariable(initializerName, kernelData, assignOpName)
}


private fun convertToLayer(kerasLayer: KerasLayer): Layer {
    return when (kerasLayer.class_name) {
        LAYER_CONV2D -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_FLATTEN -> createFlatten(kerasLayer.config!!, kerasLayer.config.name!!)
        LAYER_MAX_POOLING_2D -> createMaxPooling2D(
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
    return when (initializer.class_name!!) {
        INITIALIZER_GLOROT_UNIFORM -> GlorotUniform(12L)
        INITIALIZER_GLOROT_NORMAL -> GlorotNormal(12L)
        INITIALIZER_ZEROS -> GlorotNormal(12L) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_ONES -> GlorotNormal(12L) // instead of real initializers, because it doesn't influence on nothing
        INITIALIZER_RANDOM_NORMAL -> GlorotNormal(seed = 12L)
        "RandomUniform" -> TruncatedNormal(seed = 12L)
        "TruncatedNormal" -> TruncatedNormal(seed = 12L)
        "VarianceScaling" -> VarianceScaling(seed = 12L)
        "Constant" -> Constant(initializer.config!!.value!!.toFloat())
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
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

fun convertPadding(padding: String): ConvPadding {
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

fun recursivePrintGroup(hdfFile: HdfFile, group: Group, level: Int) {
    var level = level
    level++

    var indent = ""

    for (i in 1..level) {
        indent += "    "
    }

    for (node in group) {
        println(indent + node.name)

        for (entry in node.attributes.entries) {
            println(entry.value)
        }

        if (node is Group) {
            recursivePrintGroup(hdfFile, node, level)
        } else {
            println(node.path)
            val dataset = hdfFile.getDatasetByPath(node.path)
            val dims = arrayOf(dataset.dimensions)
            println(dims.contentDeepToString())

            when (dataset.dimensions.size) {
                4 -> {
                    val data = dataset.data as Array<Array<Array<FloatArray>>>
                    //println(data.contentDeepToString())
                }
                3 -> {
                    val data = dataset.data as Array<Array<FloatArray>>
                    //println(data.contentDeepToString())
                }
                2 -> {
                    val data = dataset.data as Array<FloatArray>
                    //println(data.contentDeepToString())
                }
                1 -> {
                    val data = dataset.data as FloatArray
                    //println(data.contentToString())
                }
            }
        }
    }
}

/*fun saveModelWeights(model: Sequential<Float>, hdfFile: HdfFile) {
    model.layers.forEach {
        run {
            when (it::class) {
                Dense::class -> writeDenseVariables(it.name, hdfFile, model)
                Conv2D::class -> writeConv2DVariables(it.name, hdfFile, model)
                else -> println("No weights loading for ${it.name}")
            }

        }
    }
}

fun writeConv2DVariables(name: String, hdfFile: HdfFile, model: Sequential<Float>) {
    TODO("Not yet implemented")
}

fun writeDenseVariables(name: String, hdfFile: HdfFile, model: Sequential<Float>) {
    val kernelData = hdfFile.getDatasetByPath("/$name/$name/kernel:0").data

    val biasData = hdfFile.getDatasetByPath("/$name/$name/bias:0").data

    val kernelVariableName = name + "_" + "dense_kernel" // TODO: to conventions
    val biasVariableName = name + "_" + "dense_bias" // TODO: to conventions
}*/



