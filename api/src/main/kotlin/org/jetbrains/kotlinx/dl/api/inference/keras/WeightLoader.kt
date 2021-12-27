/*
 * Copyright 2020-2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import io.jhdf.api.Group
import io.jhdf.api.Node
import io.jhdf.dataset.DatasetBase
import org.jetbrains.kotlinx.dl.api.core.Functional
import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.inference.keras.WeightMappings.BIAS_DATA_PATH_TEMPLATE
import org.jetbrains.kotlinx.dl.api.inference.keras.WeightMappings.KERNEL_DATA_PATH_TEMPLATE
import org.jetbrains.kotlinx.dl.api.inference.keras.WeightMappings.getLayerVariableNames
import org.jetbrains.kotlinx.dl.api.inference.keras.WeightMappings.getLayerVariablePathTemplates
import org.jetbrains.kotlinx.dl.api.inference.keras.WeightMappings.getLayerVariables


/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of the model.
 */
public fun GraphTrainableModel.loadWeights(
    hdfFile: HdfFile
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, null)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, null)
        }
        else -> {
            logger.error { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    logger.info { "Weights are loaded." }
    isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 */
public fun GraphTrainableModel.loadWeights(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }

    when {
        hdfFile.attributes.containsKey("layer_names") -> loadWeightsFromHdf5Group(hdfFile, this, layerList)
        hdfFile.children.containsKey("model_weights") -> {
            loadWeightsFromHdf5Group((hdfFile as Group).getChild("model_weights") as Group, this, layerList)
        }
        else -> {
            logger.error { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
        }
    }

    logger.info { "Weights are loaded." }
    isModelInitialized = true
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 */
public fun GraphTrainableModel.loadWeightsForFrozenLayers(
    hdfFile: HdfFile
) {
    val frozenLayers = mutableListOf<Layer>()
    layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    loadWeights(hdfFile, frozenLayers)
}

private fun loadWeightsFromHdf5Group(group: Group, model: GraphTrainableModel, layerList: MutableList<Layer>?) {
    if (group.getKerasVersion() == 1) {
        throw UnsupportedOperationException(
            "The weights loading from Keras 1.x is not supported by default!" +
                    "\nUse loadWeightsViaPathTemplates() method to make custom loading!"
        )
    }

    if (layerList != null) {
        model.layers.forEach {
            if (layerList.contains(it)) {
                fillLayerWeights(it, group, model)
            } else {
                initLayerWeights(it, model)
            }
        }
    } else {
        model.layers.forEach {
            fillLayerWeights(it, group, model)
        }
    }
}

private fun Group.getKerasVersion(): Int {
    val kerasVersionAttribute = attributes["keras_version"] ?: return 1
    if ((kerasVersionAttribute.data as String).startsWith("2")) return 2
    return 1
}

private fun fillLayerWeights(
    layer: Layer,
    group: Group,
    model: GraphTrainableModel
) {
    val variables = getLayerVariables(layer)
    if (variables == null) {
        model.logger.warn { "Loading weights for the layer ${layer.name} is skipped as ${layer::class.qualifiedName} layers are not supported." }
        return
    }
    fillLayerVariablesFromKeras(layer.name, variables, model, group)
    model.logger.debug { "${layer.paramCount} parameters loaded for the layer ${layer.name}." }
}

private fun fillLayerVariablesFromKeras(layerName: String,
                                        variables: Map<String, Pair<String, LongArray>>,
                                        model: GraphTrainableModel,
                                        group: Group
) {
    val layerWeightsNode = group.children[layerName] as? Group
    check(layerWeightsNode != null) {
        val availableLayerNames = group.children.values.map(Node::getName)
        val modelLayerNames = model.layers.map(Layer::name)
        "Weights for the loaded layer $layerName are not found in .h5 file! " +
                "\nh5 weight file contains weights for the following list of layers: $availableLayerNames" +
                "\nDouble-check your loaded configuration which contains layers with the following names: $modelLayerNames."
    }

    val nameOfWeightSubGroup = layerWeightsNode.children.keys.first()
    val dataNodes = (layerWeightsNode.children[nameOfWeightSubGroup] as Group).children

    dataNodes.values.map { it as DatasetBase }.forEach {
        val (name, shape) = variables[it.name]
            ?: throw IllegalArgumentException(
                "Parsing of h5 file for variable with name ${it.name} in layer $layerName is not supported!"
            )
        val dims = it.dimensions
        require(shape.map(Long::toInt).toIntArray().contentEquals(dims)) {
            "$name variable shape in loaded data is ${dims.contentToString()}. Should be ${shape.contentToString()}"
        }
        model.fillVariable(name, it.data)
    }
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun GraphTrainableModel.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE, // TODO: doesnt' work for batchnorm/depthwise
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }
    layers.forEach {
        fillLayerWeights(
            it,
            hdfFile,
            LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
            this
        ) // TODO: doesnt' work for batchnorm/depthwise
    }
    logger.info { "Weights are loaded." }
    isModelInitialized = true
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
public fun GraphTrainableModel.loadWeightsByPathTemplates(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }
    layers.forEach {
        if (layerList.contains(it)) {
            fillLayerWeights(
                it,
                hdfFile,
                LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                this
            ) // TODO: doesnt' work for batchnorm/depthwise
        } else {
            initLayerWeights(it, this)
        }
    }
    logger.info { "Weights are loaded." }
    isModelInitialized = true
}

private fun fillLayerWeights(
    layer: Layer,
    hdfFile: HdfFile,
    layerPaths: LayerPaths?,
    model: GraphTrainableModel
) {
    val variables = getLayerVariablePathTemplates(layer, layerPaths)
    if (variables == null) {
        model.logger.warn { "Loading weights for the layer ${layer.name} is skipped as ${layer::class.qualifiedName} layers are not supported." }
        return
    }
    variables.forEach { (variableName, variableDataPathTemplate) ->
        val data = hdfFile.getDatasetByPath(variableDataPathTemplate.format(layer.name, layer.name)).data
        model.fillVariable(variableName, data)
    }
    model.logger.debug { "${layer.paramCount} parameters loaded for the layer ${layer.name}." }
}

private fun initLayerWeights(layer: Layer, model: GraphTrainableModel) {
    val variables = getLayerVariableNames(layer)
    if (variables == null) {
        model.logger.warn { "Initializing weights for the layer ${layer.name} is skipped as ${layer::class.qualifiedName} layers are not supported." }
        return
    }
    variables.forEach(model::runAssignOpByVarName)
    model.logger.debug { "${layer.paramCount} parameters initialized for the layer ${layer.name}." }
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
public fun Functional.loadWeightsForFrozenLayersByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    val frozenLayers = mutableListOf<Layer>()
    layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    loadWeightsByPathTemplates(hdfFile, frozenLayers, kernelDataPathTemplate, biasDataPathTemplate)
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [weightPaths] Fully-specified paths to kernel and bias weights of each layer.
 *
 * NOTE: Kernel and bias will be initialized by default initializers if they are missed in [weightPaths] object.
 */
public fun GraphTrainableModel.loadWeightsByPaths(
    hdfFile: HdfFile,
    weightPaths: List<LayerPaths>,
    missedWeights: MissedWeightsStrategy = MissedWeightsStrategy.INITIALIZE,
    forFrozenLayersOnly: Boolean = false // TODO: probably it should be a flag in all methods
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }

    var layersToLoad = layers
    var layersToInit = layers

    if (forFrozenLayersOnly) {
        layersToLoad = layersToLoad.filter { !it.isTrainable }
        layersToInit = layersToInit.filter { it.isTrainable }
        layersToInit.forEach {
            initLayerWeights(it, this)
        }
    }

    layersToLoad.forEach {
        val initializedLayerName = it.name
        val layerWeightPaths = weightPaths.find { initializedLayerName == it.layerName }
        if (layerWeightPaths != null) {
            fillLayerWeights(it, hdfFile, layerWeightPaths, this)
        } else {
            if (missedWeights == MissedWeightsStrategy.LOAD_CUSTOM_PATH) {
                fillLayerWeights(
                    it,
                    hdfFile,
                    null, // TODO: refactor = it doesn't work for batchnorm or depthwise
                    this
                )
            } else {
                logger.warn { "Layer weight paths for ${it.name} are not found in 'weightPaths' object. It will be initialized by default initializer." }
                initLayerWeights(it, this)
            }
        }
    }

    logger.info { "Weights are loaded." }
    isModelInitialized = true // TODO: it should depend on what is happened with missed weights
}

/** This strategy defines the behaviour during weights' loading if the weights are not found in the h5 file by the standard Keras paths. */
public enum class MissedWeightsStrategy {
    /** In this case the missed weights should be filled via initializer. */
    INITIALIZE,

    /** In this case the loader should try to load them by the alternative path proposed by the user. */
    LOAD_CUSTOM_PATH
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
public fun GraphTrainableModel.loadWeightsByPaths(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    logger.info { "Starting weights loading.." }
    layers.forEach {
        if (layerList.contains(it)) {
            fillLayerWeights(
                it,
                hdfFile,
                LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                this
            ) // TODO: does not work for BatchNorm/Depthwise
        } else {
            initLayerWeights(it, this)
        }
    }
    logger.info { "Weights are loaded." }
    isModelInitialized = true
}