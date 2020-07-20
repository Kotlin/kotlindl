package examples.experimental.hdf5


import api.defaultAssignOpName
import api.defaultInitializerOpName
import api.keras.Sequential
import api.keras.activations.Activations
import api.keras.dataset.ImageDataset
import api.keras.initializers.*
import api.keras.layers.Dense
import api.keras.layers.Flatten
import api.keras.layers.Input
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import api.keras.layers.twodim.ConvPadding
import api.keras.layers.twodim.MaxPool2D
import api.keras.loss.LossFunctions
import api.keras.metric.Metrics
import api.keras.optimizers.Adam
import com.beust.klaxon.Klaxon
import examples.experimental.hdf5.lenetconfig.*
import examples.keras.mnist.util.*
import examples.production.getLabel
import io.jhdf.HdfFile
import io.jhdf.api.Group
import java.io.File

private

fun main() {
    val pathToConfig = "models/mnist/lenet/model.json"
    val realPathToConfig = ImageDataset::class.java.classLoader.getResource(pathToConfig).path.toString()

    val jsonConfigFile = File(realPathToConfig)
    val jsonString = jsonConfigFile.readText(Charsets.UTF_8)
    println(jsonString)

    val sequentialConfig = Klaxon()
        .parse<SequentialConfig>(jsonString)

    println(sequentialConfig.toString())


    /* hdfFile.use { hdfFile ->
         recursivePrintGroup(hdfFile, hdfFile, 0)
     }*/

    val model = buildSequentialModelByJSONConfig(sequentialConfig!!)
    model.compile(
        optimizer = Adam<Float>(),
        loss = LossFunctions.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
        metric = Metrics.ACCURACY
    )
    model.summary()


    val jsonConfigFile2 = File("savedmodels/mnisSeq.json")


    saveSequentialModelToJSONConfig(model, jsonConfigFile2)


    val pathToWeights = "models/mnist/lenet/mnist_weights_only.h5"
    val realPathToWeights = ImageDataset::class.java.classLoader.getResource(pathToWeights).path.toString()

    val file = File(realPathToWeights)
    println(file.isFile)

    val hdfFile = HdfFile(file)

    loadWeightsToModel(model, hdfFile)


    val pathToWeightsForPython = "savedmodels/mnist_weights_only.h5"
    val file2 = File(pathToWeightsForPython)
    val hdfFile2 = HdfFile(file2)

    saveModelWeights(model, hdfFile2)


    val (train, test) = ImageDataset.createTrainAndTestDatasets(
        TRAIN_IMAGES_ARCHIVE,
        TRAIN_LABELS_ARCHIVE,
        TEST_IMAGES_ARCHIVE,
        TEST_LABELS_ARCHIVE,
        AMOUNT_OF_CLASSES,
        ::extractImages,
        ::extractLabels
    )

    val imageId1 = 0
    val imageId2 = 1
    val imageId3 = 2

    model.use {
        val prediction = it.predict(train.getImage(imageId1))

        println("Prediction: $prediction Ground Truth: ${getLabel(train, imageId1)}")

        val prediction2 = it.predict(train.getImage(imageId2))

        println("Prediction: $prediction2 Ground Truth: ${getLabel(train, imageId2)}")

        val prediction3 = it.predict(train.getImage(imageId3))

        println("Prediction: $prediction3 Ground Truth: ${getLabel(train, imageId3)}")
    }

}

fun saveModelWeights(model: Sequential<Float>, hdfFile: HdfFile) {
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
}

fun saveSequentialModelToJSONConfig(model: Sequential<Float>, jsonConfigFile: File) {
    val kerasLayers = mutableListOf<KerasLayer>()
    model.layers.forEach {
        run {
            val layer = convertToKerasLayer(it)
            kerasLayers.add(layer)
        }
    }

    val inputShape = model.firstLayer.packedDims.map { it.toInt() }
    (kerasLayers.first().config as ConfigX).batch_input_shape =
        listOf(null, inputShape[0], inputShape[1], inputShape[2])

    val config = Config(name = "", layers = kerasLayers)
    val sequentialConfig = SequentialConfig(config = config)

    val jsonString2 = Klaxon().toJsonString(sequentialConfig)

    jsonConfigFile.writeText(jsonString2, Charsets.UTF_8)
}

fun convertToKerasLayer(layer: Layer<Float>): KerasLayer {
    return when (layer::class) {
        Conv2D::class -> createKerasConv2D(layer as Conv2D<Float>)
        Flatten::class -> createKerasFlatten(layer as Flatten<Float>)
        MaxPool2D::class -> createKerasMaxPooling2D(layer as MaxPool2D<Float>)
        Dense::class -> createKerasDense(layer as Dense<Float>)
        else -> throw IllegalStateException("${layer.name} is not supported yet!")
    }
}

fun createKerasDense(layer: Dense<Float>): KerasLayer {
    val configX = ConfigX(
        dtype = "float32",
        units = layer.outputSize,
        name = layer.name,
        use_bias = true,
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasKernelInitializer(layer.kernelInitializer),
        bias_initializer = convertToKerasBiasInitializer(layer.biasInitializer)
    )
    return KerasLayer(class_name = "Dense", config = configX)
}

fun convertToKerasBiasInitializer(biasInitializer: Initializer<Float>): BiasInitializer? {
    val className = when (biasInitializer::class) {
        GlorotUniform::class -> "GlorotUniform"
        GlorotNormal::class -> "GlorotNormal"
        Zeros::class -> "Zeros"
        Ones::class -> "Ones"
        else -> throw IllegalStateException("${biasInitializer::class.simpleName} is not supported yet!")
    }
    val config = ConfigXXX(seed = 12)
    return BiasInitializer(class_name = className, config = config)
}

fun convertToKerasKernelInitializer(kernelInitializer: Initializer<Float>): KernelInitializer? {
    val className = when (kernelInitializer::class) {
        GlorotUniform::class -> "GlorotUniform"
        GlorotNormal::class -> "GlorotNormal"
        Zeros::class -> "Zeros"
        Ones::class -> "Ones"
        else -> throw IllegalStateException("${kernelInitializer::class.simpleName} is not supported yet!")
    }
    val config = ConfigXXXXX(seed = 12)
    return KernelInitializer(class_name = className, config = config)
}

fun convertToKerasActivation(activation: Activations): String? {
    return when (activation) {
        Activations.Relu -> "relu"
        Activations.Sigmoid -> "sigmoid"
        Activations.Softmax -> "softmax"
        Activations.Linear -> "linear"
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

fun createKerasMaxPooling2D(layer: MaxPool2D<Float>): KerasLayer {
    val poolSize = mutableListOf(layer.poolSize[1], layer.poolSize[2])
    val strides = mutableListOf(layer.strides[1], layer.strides[2])
    val configX = ConfigX(
        data_format = "channels_last",
        dtype = "float32",
        name = layer.name,
        padding = "same",
        pool_size = poolSize,
        strides = strides
    )
    return KerasLayer(class_name = "MaxPooling2D", config = configX)
}

fun createKerasFlatten(layer: Flatten<Float>): KerasLayer {
    val configX = ConfigX(data_format = "channels_last", dtype = "float32", name = layer.name)
    return KerasLayer(class_name = "Flatten", config = configX)
}

fun createKerasConv2D(layer: Conv2D<Float>): KerasLayer {
    val kernelSize = layer.kernelSize.map { it.toInt() }.toList()
    val configX = ConfigX(
        filters = layer.filters.toInt(),
        kernel_size = kernelSize,
        strides = listOf(layer.strides[1].toInt(), layer.strides[2].toInt()),
        activation = convertToKerasActivation(layer.activation),
        kernel_initializer = convertToKerasKernelInitializer(layer.kernelInitializer),
        bias_initializer = convertToKerasBiasInitializer(layer.biasInitializer),
        padding = "same",
        name = layer.name,
        use_bias = true
    )
    return KerasLayer(class_name = "Conv2D", config = configX)
}

fun loadWeightsToModel(model: Sequential<Float>, hdfFile: HdfFile) {
    println(model.kGraph.toString())

    model.layers.forEach {
        run {
            when (it::class) {
                Dense::class -> fillDenseVariables(it.name, hdfFile, model)
                Conv2D::class -> fillConv2DVariables(it.name, hdfFile, model)
                else -> println("No weights loading for ${it.name}")
            }

        }
    }
}

fun buildSequentialModelByJSONConfig(
    sequentialConfig: SequentialConfig
): Sequential<Float> {
    val layers = mutableListOf<Layer<Float>>()

    sequentialConfig.config!!.layers!!.forEach {
        run {
            val layer = convertToLayer(it)
            layers.add(layer)
        }
    }

    val input = Input<Float>(28, 28, 1)

    return Sequential.of(input, layers.toList())
}

fun fillConv2DVariables(name: String, hdfFile: HdfFile, model: Sequential<Float>) {
    val kernelData = hdfFile.getDatasetByPath("/$name/$name/kernel:0").data
    val biasData = hdfFile.getDatasetByPath("/$name/$name/bias:0").data

    val kernelVariableName = name + "_" + "conv2d_kernel" // TODO: to conventions
    val biasVariableName = name + "_" + "conv2d_bias" // TODO: to conventions
    addInitOpsToGraph(kernelVariableName, model, kernelData)
    addInitOpsToGraph(biasVariableName, model, biasData)
}

fun fillDenseVariables(name: String, hdfFile: HdfFile, model: Sequential<Float>) {
    val kernelData = hdfFile.getDatasetByPath("/$name/$name/kernel:0").data
    val biasData = hdfFile.getDatasetByPath("/$name/$name/bias:0").data

    val kernelVariableName = name + "_" + "dense_kernel" // TODO: to conventions
    val biasVariableName = name + "_" + "dense_bias" // TODO: to conventions
    addInitOpsToGraph(kernelVariableName, model, kernelData)
    addInitOpsToGraph(biasVariableName, model, biasData)
}

private fun addInitOpsToGraph(
    variableName: String,
    model: Sequential<Float>,
    kernelData: Any
) {
    val initializerName = defaultInitializerOpName(variableName)
    val assignOpName = defaultAssignOpName(variableName)

    model.populateVariable(initializerName, kernelData, assignOpName)
}


fun convertToLayer(kerasLayer: KerasLayer): Layer<Float> {
    return when (kerasLayer.class_name) {
        "Conv2D" -> createConv2D(kerasLayer.config!!, kerasLayer.config.name!!)
        "Flatten" -> createFlatten(kerasLayer.config!!, kerasLayer.config.name!!)
        "MaxPooling2D" -> createMaxPooling2D(kerasLayer.config!!, kerasLayer.config.name!!)
        "Dense" -> createDense(kerasLayer.config!!, kerasLayer.config.name!!)
        else -> throw IllegalStateException("${kerasLayer.config!!.name!!} is not supported yet!")
    }
}

fun createDense(config: ConfigX, name: String): Dense<Float> {
    return Dense(
        outputSize = config.units!!,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToKernelInitializer(config.kernel_initializer!!),
        biasInitializer = convertToBiasInitializer(config.bias_initializer!!),
        name = name
    )
}

fun convertToBiasInitializer(initializer: BiasInitializer): Initializer<Float> {
    return when (initializer.class_name!!) {
        "GlorotUniform" -> GlorotUniform(12L)
        "GlorotNormal" -> GlorotNormal(12L)
        "Zeros" -> GlorotUniform(12L) // instead of real initializers, because it doesn't influence on nothing
        "Ones" -> GlorotUniform(12L) // instead of real initializers, because it doesn't influence on nothing
        "RandomNormal" -> GlorotNormal(seed = 12L)
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
    }
}

fun convertToKernelInitializer(initializer: KernelInitializer): Initializer<Float> {
    return when (initializer.class_name!!) {
        "GlorotUniform" -> GlorotUniform(12L)
        "GlorotNormal" -> GlorotNormal(12L)
        "Zeros" -> GlorotUniform(12L) // instead of real initializers, because it doesn't influence on nothing
        "Ones" -> GlorotUniform(12L) // instead of real initializers, because it doesn't influence on nothing
        "RandomNormal" -> GlorotNormal(seed = 12L)
        else -> throw IllegalStateException("${initializer.class_name} is not supported yet!")
    }
}

fun convertToActivation(activation: String): Activations {
    return when (activation) {
        "relu" -> Activations.Relu
        "sigmoid" -> Activations.Sigmoid
        "softmax" -> Activations.Softmax
        "linear" -> Activations.Linear
        else -> throw IllegalStateException("$activation is not supported yet!")
    }
}

fun createMaxPooling2D(config: ConfigX, name: String): MaxPool2D<Float> {
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

    return MaxPool2D<Float>(addedOnesPoolSize, addedOnesStrides)
}

fun createFlatten(config: ConfigX, name: String): Flatten<Float> {
    return Flatten()
}

fun createConv2D(config: ConfigX, name: String): Conv2D<Float> {
    val kernelSize = config.kernel_size!!.map { it.toLong() }.toLongArray()
    val strides = config.strides!!.map { it.toLong() }.toLongArray()

    val addedOnesStrides = LongArray(4)
    addedOnesStrides[0] = 1
    addedOnesStrides[1] = strides[0]
    addedOnesStrides[2] = strides[1]
    addedOnesStrides[3] = 1

    return Conv2D(
        filters = config.filters!!.toLong(),
        kernelSize = kernelSize,
        strides = addedOnesStrides,
        activation = convertToActivation(config.activation!!),
        kernelInitializer = convertToKernelInitializer(config.kernel_initializer!!),
        biasInitializer = convertToBiasInitializer(config.bias_initializer!!),
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





