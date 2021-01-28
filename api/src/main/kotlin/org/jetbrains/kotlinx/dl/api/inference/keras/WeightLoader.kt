/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import io.jhdf.api.Group
import io.jhdf.dataset.DatasetBase
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D
import org.jetbrains.kotlinx.dl.api.core.util.*

private const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
private const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"


/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 */
public fun Sequential.loadWeights(
    hdfFile: HdfFile
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, null)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, null)
        }
        else -> {
            this.logger.info { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 */
public fun Sequential.loadWeights(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, layerList)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, layerList)
        }
        else -> {
            this.logger.info { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 */
public fun Sequential.loadWeightsForFrozenLayers(
    hdfFile: HdfFile
) {
    val frozenLayers = mutableListOf<Layer>()
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeights(hdfFile, frozenLayers)
}

private fun loadWeightsFromHdf5Group(group: Group, model: Sequential, layerList: MutableList<Layer>?) {
    var originalKerasVersion = 1

    if (group.attributes.containsKey("keras_version") && ((if (group.attributes["keras_version"] != null) group.attributes["keras_version"]?.data else "1") as String).startsWith(
            "2"
        )
    ) {
        originalKerasVersion = 2
    }
    if (originalKerasVersion == 1
    ) {
        throw UnsupportedOperationException(
            "The weights loading from Keras 1.x is not supported by default!" +
                    "\nUse loadWeightsViaPathTemplates() method to make custom loading!"
        )
    }

    if (layerList != null) {
        model.layers.forEach {
            run {
                if (layerList.contains(it)) {
                    fillLayerWeights(it, group, model)
                } else {
                    initLayerWeights(it, model)
                }
            }
        }
    } else {
        model.layers.forEach {
            run {
                fillLayerWeights(it, group, model)
            }
        }
    }
}

private fun fillLayerWeights(
    it: Layer,
    group: Group,
    model: Sequential
) {
    when (it::class) {
        Dense::class -> fillDenseVariablesFromKeras(it.name, group, model)
        Conv2D::class -> fillConv2DVariablesFromKeras(
            it.name,
            group,
            model
        )
        BatchNorm::class -> fillBatchNormVariablesFromKeras(it.name, group, model)
        else -> println("No weights loading for ${it.name}")
    }
    model.logger.info { " Weights loaded for ${it.name}. ${it.paramCount} parameters are loaded. " }
}

private fun fillConv2DVariablesFromKeras(
    layerName: String,
    group: Group,
    model: Sequential
) {
    val availableLayerNames = group.children.map { e -> group.children[e.key]!!.name }.toList()
    val modelLayerNames = model.layers.map { e -> e.name }.toList()
    val layerWeightsNode = group.children[layerName]
    check(layerWeightsNode != null) {
        "Weights for the loaded Conv2D layer $layerName are not found in .h5 file! " +
                "\nh5 weight file contains weights for the following list of layers: $availableLayerNames" +
                "\nDouble-check your loaded configuration which contains layers with the following names: $modelLayerNames."
    }

    val firstLevelGroup: Group = layerWeightsNode as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "kernel:0" -> {
                val kernelVariableName = conv2dKernelVarName(layerName)
                val kernelShape = (model.getLayer(layerName) as Conv2D).kernelShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "bias:0" -> {
                val biasVariableName = conv2dBiasVarName(layerName)
                val biasShape = (model.getLayer(layerName) as Conv2D).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

private fun fillDenseVariablesFromKeras(
    layerName: String,
    group: Group,
    model: Sequential
) {
    val firstLevelGroup: Group = group.children[layerName] as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            "kernel:0" -> {
                val kernelVariableName = denseKernelVarName(layerName)
                val kernelShape = (model.getLayer(layerName) as Dense).kernelShapeArray
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }
            "bias:0" -> {
                val biasVariableName = denseBiasVarName(layerName)
                val biasShape = (model.getLayer(layerName) as Dense).biasShapeArray
                require(
                    biasShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${biasShape.contentToString()}" }
                model.fillVariable(biasVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }
    }
}

private fun fillBatchNormVariablesFromKeras(
    layerName: String,
    group: Group,
    model: Sequential
) {
    val firstLevelGroup: Group = group.children[layerName] as Group
    val nameOfWeightSubGroup = firstLevelGroup.children.keys.first()
    val dataNodes = (firstLevelGroup.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val dims = it.dimensions
        val data = it.data
        when (it.name) {
            /*"kernel:0" -> {
                val kernelVariableName = denseKernelVarName(layerName)
                val kernelShape = (model.getLayer(layerName) as Dense).getKernelShape()
                require(
                    kernelShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Kernel shape in loaded data is ${dims.contentToString()}. Should be ${kernelShape.contentToString()}" }
                model.fillVariable(kernelVariableName, data)
            }*/
            "gamma:0" -> {
                val gammaVariableName = batchNormGammaVarName(layerName)
                val gammaShape = (model.getLayer(layerName) as BatchNorm).getWeightShape()
                require(
                    gammaShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Gamma shape in loaded data is ${dims.contentToString()}. Should be ${gammaShape.contentToString()}" }
                model.fillVariable(gammaVariableName, data)
            }
            "beta:0" -> {
                val betaVariableName = batchNormBetaVarName(layerName)
                val betaShape = (model.getLayer(layerName) as BatchNorm).getWeightShape()
                require(
                    betaShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Beta shape in loaded data is ${dims.contentToString()}. Should be ${betaShape.contentToString()}" }
                model.fillVariable(betaVariableName, data)
            }
            "moving_mean:0" -> {
                val movingMeanVariableName = batchNormMovingMeanVarName(layerName)
                val movingMeanShape = (model.getLayer(layerName) as BatchNorm).getWeightShape()
                require(
                    movingMeanShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Moving mean shape in loaded data is ${dims.contentToString()}. Should be ${movingMeanShape.contentToString()}" }
                model.fillVariable(movingMeanVariableName, data)
            }
            "moving_variance:0" -> {
                val movingVarianceVariableName = batchNormMovingVarianceVarName(layerName)
                val movingVarianceShape = (model.getLayer(layerName) as BatchNorm).getWeightShape()
                require(
                    movingVarianceShape.map { e -> e.toInt() }.toIntArray().contentEquals(dims)
                ) { "Moving variance shape in loaded data is ${dims.contentToString()}. Should be ${movingVarianceShape.contentToString()}" }
                model.fillVariable(movingVarianceVariableName, data)
            }
            else -> {
                throw IllegalArgumentException("Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!")
            }
        }

    }
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            fillLayerWeights(it, hdfFile, kernelDataPathTemplate, biasDataPathTemplate, this)
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (layerList.contains(it)) {
                fillLayerWeights(it, hdfFile, kernelDataPathTemplate, biasDataPathTemplate, this)
            } else {
                initLayerWeights(it, this)
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

private fun fillLayerWeights(
    it: Layer,
    hdfFile: HdfFile,
    kernelDataPathTemplate: String,
    biasDataPathTemplate: String,
    model: Sequential
) {
    when (it::class) {
        Dense::class -> fillDenseVariables(
            it.name,
            hdfFile,
            model,
            kernelDataPathTemplate,
            biasDataPathTemplate
        )
        Conv2D::class -> fillConv2DVariables(
            it.name,
            hdfFile,
            model,
            kernelDataPathTemplate,
            biasDataPathTemplate
        )
        else -> println("No weights loading for ${it.name}")
    }
    model.logger.info { " Weights loaded for ${it.name}. ${it.paramCount} parameters are loaded. " }
}

private fun initLayerWeights(it: Layer, model: Sequential) {
    when (it::class) {
        Dense::class -> initDenseVariablesByDefaultInitializer(it.name, model)
        Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, model)
        else -> println("No default initialization handled for ${it.name}")
    }
    model.logger.info { " Weights initialized for ${it.name}. ${it.paramCount} parameters are initialized. " }
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeightsForFrozenLayersByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    val frozenLayers = mutableListOf<Layer>()
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeightsByPathTemplates(hdfFile, frozenLayers, kernelDataPathTemplate, biasDataPathTemplate)
}

private fun initConv2DVariablesByDefaultInitializer(name: String, model: Sequential) {
    val kernelVariableName = conv2dKernelVarName(name)
    val biasVariableName = conv2dBiasVarName(name)
    model.runAssignOpByVarName(kernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

private fun initDenseVariablesByDefaultInitializer(name: String, model: Sequential) {
    val kernelVariableName = denseKernelVarName(name)
    val biasVariableName = denseBiasVarName(name)
    model.runAssignOpByVarName(kernelVariableName)
    model.runAssignOpByVarName(biasVariableName)
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [weightPaths] Fully-specified paths to kernel and bias weights of each layer.
 *
 * NOTE: Kernel and bias will be initialized by default initializers if they are missed in [weightPaths] object.
 */
public fun Sequential.loadWeightsByPaths(
    hdfFile: HdfFile,
    weightPaths: List<LayerKernelAndBiasPaths>
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            val initializedLayerName = it.name
            val layerWeightPaths = weightPaths.find { initializedLayerName == it.layerName }
            if (layerWeightPaths != null) {
                val kernelDataPathTemplate = layerWeightPaths.kernelPath
                val biasDataPathTemplate = layerWeightPaths.biasPath
                fillLayerWeights(it, hdfFile, kernelDataPathTemplate, biasDataPathTemplate, this)
            } else {
                this.logger.info { "Layer weight paths for ${it.name} are not found in 'weightPaths' object. It will be initialized by default initializer." }
                initLayerWeights(it, this)
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeightsByPaths(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            if (layerList.contains(it)) {
                fillLayerWeights(it, hdfFile, kernelDataPathTemplate, biasDataPathTemplate, this)
            } else {
                initLayerWeights(it, this)
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

/**
 * Contains [layerName], [kernelPath], [biasPath] for specific layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public data class LayerKernelAndBiasPaths(
    /** */
    public val layerName: String,
    /** */
    public val kernelPath: String,
    /** */
    public val biasPath: String
)

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
    model.fillVariable(kernelVariableName, kernelData)
    model.fillVariable(biasVariableName, biasData)
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

    model.fillVariable(kernelVariableName, kernelData)
    model.fillVariable(biasVariableName, biasData)
}
