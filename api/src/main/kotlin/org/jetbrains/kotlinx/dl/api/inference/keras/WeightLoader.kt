/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import io.jhdf.api.Group
import io.jhdf.api.Node
import io.jhdf.dataset.DatasetBase
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
public fun GraphTrainableModel.loadWeights(hdfFile: HdfFile): Unit = loadWeights(hdfFile, layers)

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 */
public fun GraphTrainableModel.loadWeightsForFrozenLayers(hdfFile: HdfFile) {
    loadWeights(hdfFile, layers.filterNot(Layer::isTrainable))
}

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for pre-defined list of layers.
 *
 * NOTE: Weights for another layers will not be loaded (should be initialized manually).
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [layerList] List of layers to load weights. Weights for other layers will be initialized by initializer later.
 */
public fun GraphTrainableModel.loadWeights(hdfFile: HdfFile, layerList: List<Layer>) {
    val group = when {
        hdfFile.attributes.containsKey("layer_names") -> hdfFile
        hdfFile.children.containsKey("model_weights") -> (hdfFile as Group).getChild("model_weights") as Group
        else -> null
    }
    if (group == null) {
        logger.error {
            "This is unknown path format. Use special method loadWeightsViaPathTemplates()" +
                    " to specify templates to load weights."
        }
        return
    }

    if (group.getKerasVersion() == 1) {
        throw UnsupportedOperationException(
            "The weights loading from Keras 1.x is not supported by default!" +
                    "\nUse loadWeightsViaPathTemplates() method to make custom loading!"
        )
    }

    loadWeights(layerList) { layer -> fillLayerWeights(layer, group, this) }
}

private fun Group.getKerasVersion(): Int {
    val kerasVersionAttribute = attributes["keras_version"] ?: return 1
    if ((kerasVersionAttribute.data as String).startsWith("2")) return 2
    return 1
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
): Unit = loadWeightsByPathTemplates(hdfFile, layers, kernelDataPathTemplate, biasDataPathTemplate)

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework for non-trainable (or frozen) layers only.
 *
 * NOTE: Weights for trainable layers will not be loaded and will be initialized via default initializers.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun GraphTrainableModel.loadWeightsForFrozenLayersByPathTemplates(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    loadWeightsByPathTemplates(
        hdfFile, layers.filterNot(Layer::isTrainable),
        kernelDataPathTemplate, biasDataPathTemplate
    )
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
    layerList: List<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    // TODO: doesnt' work for batchnorm/depthwise
    val layerPaths = LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate)
    loadWeights(layerList) { layer ->
        fillLayerWeights(layer, hdfFile, layerPaths, this)
    }
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
    val layersToLoad = if (forFrozenLayersOnly) layers.filterNot(Layer::isTrainable) else layers

    val layersToWeightPaths = layersToLoad.mapNotNull { layer ->
        val layerPaths = weightPaths.find { layer.name == it.layerName }
        if (layerPaths == null && missedWeights == MissedWeightsStrategy.INITIALIZE) {
            logger.warn {
                "Layer weight paths for ${layer.name} are not found in 'weightPaths' object." +
                        " Initialization is going to be done by default initializer."
            }
            return@mapNotNull null
        }
        layer to layerPaths
        // TODO: refactor when weight path is not provided and strategy is not INITIALIZE it won't work for batchnorm or depthwise
    }.toMap()

    loadWeights(layersToWeightPaths.keys) { layer ->
        fillLayerWeights(layer, hdfFile, layersToWeightPaths[layer], this)
    }
    // TODO: isModelInitialized should depend on what is happened with missed weights
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
    layerList: List<Layer>,
    kernelDataPathTemplate: String = KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = BIAS_DATA_PATH_TEMPLATE
) {
    // TODO: does not work for BatchNorm/Depthwise
    val layerConvOrDensePaths = LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate)
    loadWeights(layerList) { layer ->
        fillLayerWeights(layer, hdfFile, layerConvOrDensePaths, this)
    }
}

private fun GraphTrainableModel.loadWeights(layersToLoad: Collection<Layer>, loadWeightsBlock: (Layer) -> Unit) {
    check(isModelCompiled) { "Model is not compiled yet. Compile the model before loading weights." }
    check(!isModelInitialized) { "Model is already initialized." }
    logger.info { "Starting loading weights..." }

    val layerSet = layersToLoad.toSet()
    layers.forEach { layer ->
        if (layerSet.contains(layer)) {
            loadWeightsBlock(layer)
        } else {
            initLayerWeights(layer, this)
        }
    }

    logger.info { "Weights are loaded." }
    isModelInitialized = true
}

private fun fillLayerWeights(layer: Layer, group: Group, model: GraphTrainableModel) {
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

private fun fillLayerWeights(layer: Layer, hdfFile: HdfFile, layerPaths: LayerPaths?, model: GraphTrainableModel) {
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