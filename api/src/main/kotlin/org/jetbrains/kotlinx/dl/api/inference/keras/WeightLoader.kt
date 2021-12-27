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
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.DepthwiseConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.SeparableConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.util.*

private const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
private const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"
private const val GAMMA_DATA_PATH_TEMPLATE = "/%s/%s/gamma:0"
private const val BETA_DATA_PATH_TEMPLATE = "/%s/%s/beta:0"
private const val MOVING_MEAN_DATA_PATH_TEMPLATE = "/%s/%s/moving_mean:0"
private const val MOVING_VARIANCE_DATA_PATH_TEMPLATE = "/%s/%s/moving_variance:0"
private const val DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_kernel:0"
private const val POINTWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/pointwise_kernel:0"
private const val DEPTHWISE_BIAS_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_bias:0"

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of the model.
 */
public fun GraphTrainableModel.loadWeights(
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
            this.logger.error { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
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
public fun GraphTrainableModel.loadWeights(
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
            this.logger.error { "This is unknown path format. Use special method loadWeightsViaPathTemplates() to specify templates to load weights." }
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
public fun GraphTrainableModel.loadWeightsForFrozenLayers(
    hdfFile: HdfFile
) {
    val frozenLayers = mutableListOf<Layer>()
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeights(hdfFile, frozenLayers)
}

private fun loadWeightsFromHdf5Group(group: Group, model: GraphTrainableModel, layerList: MutableList<Layer>?) {
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
    layer: Layer,
    group: Group,
    model: GraphTrainableModel
) {
    val variables = getLayerVariables(layer)
    if (variables == null) return
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

private fun getConv2DVariables(layer: Conv2D): Map<String, Pair<String, LongArray>> {
    val variables = mutableMapOf(
        Pair("kernel:0", Pair(convKernelVarName(layer.name, dim = 2), layer.kernelShapeArray))
    )
    if (layer.useBias) {
        variables["bias:0"] = Pair(convBiasVarName(layer.name, dim = 2), layer.biasShapeArray!!)
    }
    return variables
}

private fun getDepthwiseConv2DVariables(layer: DepthwiseConv2D): Map<String, Pair<String, LongArray>> {
    val variables = mutableMapOf(
        Pair("depthwise_kernel:0", Pair(depthwiseConv2dKernelVarName(layer.name), layer.kernelShapeArray))
    )
    if (layer.useBias) {
        variables["depthwise_bias:0"] = Pair(depthwiseConv2dBiasVarName(layer.name), layer.biasShapeArray!!)
    }
    return variables
}

private fun getSeparableConv2DVariables(layer: SeparableConv2D): Map<String, Pair<String, LongArray>> {
    val variables = mutableMapOf(
        Pair("depthwise_kernel:0", Pair(separableConv2dDepthwiseKernelVarName(layer.name), layer.depthwiseShapeArray)),
        Pair("pointwise_kernel:0", Pair(separableConv2dPointwiseKernelVarName(layer.name), layer.pointwiseShapeArray))
    )
    if (layer.useBias) {
        variables["bias:0"] = Pair(separableConv2dBiasVarName(layer.name), layer.biasShapeArray!!)
    }
    return variables
}

private fun getDenseVariables(layer: Dense): Map<String, Pair<String, LongArray>> {
    val variables = mutableMapOf(
        Pair("kernel:0", Pair(denseKernelVarName(layer.name), layer.kernelShapeArray))
    )
    if (layer.useBias) {
        variables["bias:0"] = Pair(denseBiasVarName(layer.name), layer.biasShapeArray!!)
    }
    return variables
}

private fun getBatchNormVariables(layer: BatchNorm): Map<String, Pair<String, LongArray>> {
    val variables = mutableMapOf(
        Pair("moving_mean:0", Pair(batchNormMovingMeanVarName(layer.name), layer.movingMeanShapeArray)),
        Pair("moving_variance:0", Pair(batchNormMovingVarianceVarName(layer.name), layer.movingVarianceShapeArray))
    )
    if (layer.scale) {
        variables["gamma:0"] = Pair(batchNormGammaVarName(layer.name), layer.gammaShapeArray!!)
    }
    if (layer.center) {
        variables["beta:0"] = Pair(batchNormBetaVarName(layer.name), layer.betaShapeArray!!)
    }
    return variables
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
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
            fillLayerWeights(
                it,
                hdfFile,
                LayerConvOrDensePaths("", kernelDataPathTemplate, biasDataPathTemplate),
                this
            ) // TODO: doesnt' work for batchnorm/depthwise
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
public fun GraphTrainableModel.loadWeightsByPathTemplates(
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
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

private fun fillLayerWeights(
    layer: Layer,
    hdfFile: HdfFile,
    layerPaths: LayerPaths?,
    model: GraphTrainableModel
) {
    val variables = when (layer) {
        is Dense -> getDenseVariables(layer, layerPaths)
        is Conv2D -> getConv2DVariables(layer, layerPaths)
        is DepthwiseConv2D -> getDepthwiseConv2DVariables(layer, layerPaths)
        is SeparableConv2D -> getSeparableConv2DVariables(layer, layerPaths)
        is BatchNorm -> getBatchNormVariables(layer, layerPaths)
        else -> null
    }
    if (variables == null) return
    variables.forEach { (variableName, variableDataPathTemplate) ->
        val data = hdfFile.getDatasetByPath(variableDataPathTemplate.format(layer.name, layer.name)).data
        model.fillVariable(variableName, data)
    }
    model.logger.debug { "${layer.paramCount} parameters loaded for the layer ${layer.name}." }
}

private fun initLayerWeights(layer: Layer, model: GraphTrainableModel) {
    val variables = getLayerVariableNames(layer)
    if (variables == null) return
    variables.forEach(model::runAssignOpByVarName)
    model.logger.debug { "${layer.paramCount} parameters initialized for the layer ${layer.name}." }
}

private fun getLayerVariableNames(layer: Layer): List<String>? {
    return getLayerVariables(layer)?.map { (_, variable) -> variable.first }
}

// TODO: add loading for all layers with weights from Keras like Conv1D and Conv3D
private fun getLayerVariables(layer: Layer): Map<String, Pair<String, LongArray>>? {
    val variables = when (layer) {
        is Dense -> getDenseVariables(layer)
        is Conv2D -> getConv2DVariables(layer)
        is DepthwiseConv2D -> getDepthwiseConv2DVariables(layer)
        is SeparableConv2D -> getSeparableConv2DVariables(layer)
        is BatchNorm -> getBatchNormVariables(layer)
        else -> null
    }
    return variables
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
    this.layers.forEach {
        if (!it.isTrainable) frozenLayers.add(it)
    }
    this.loadWeightsByPathTemplates(hdfFile, frozenLayers, kernelDataPathTemplate, biasDataPathTemplate)
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
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }

    var layersToLoad = this.layers
    var layersToInit = this.layers

    if (forFrozenLayersOnly) {
        layersToLoad = layersToLoad.filter { !it.isTrainable }
        layersToInit = layersToInit.filter { it.isTrainable }
        layersToInit.forEach {
            initLayerWeights(it, this)
        }
    }

    layersToLoad.forEach {
        run {
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
                    this.logger.warn { "Layer weight paths for ${it.name} are not found in 'weightPaths' object. It will be initialized by default initializer." }
                    initLayerWeights(it, this)
                }
            }
        }
    }

    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true // TODO: it should depend on what is happened with missed weights
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
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.info { "Starting weights loading.." }
    this.layers.forEach {
        run {
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
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
}

private fun getConv2DVariables(layer: Conv2D,
                               layerPaths: LayerPaths?
): Map<String, String> {
    val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
        ?: LayerConvOrDensePaths(layer.name, KERNEL_DATA_PATH_TEMPLATE, BIAS_DATA_PATH_TEMPLATE)
    val variables = mutableMapOf(
        Pair(convKernelVarName(layer.name, dim = 2), layerConvOrDensePaths.kernelPath)
    )
    if (layer.useBias) {
        variables[convBiasVarName(layer.name, dim = 2)] = layerConvOrDensePaths.biasPath
    }
    return variables
}

private fun getDepthwiseConv2DVariables(layer: DepthwiseConv2D,
                                        layerPaths: LayerPaths?
): Map<String, String> {
    val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
        ?: LayerConvOrDensePaths(
            layer.name,
            DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE,
            DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
        )
    val variables = mutableMapOf(
        Pair(depthwiseConv2dKernelVarName(layer.name), layerConvOrDensePaths.kernelPath)
    )
    if (layer.useBias) {
        variables[depthwiseConv2dBiasVarName(layer.name)] = layerConvOrDensePaths.biasPath
    }
    return variables
}

private fun getSeparableConv2DVariables(layer: SeparableConv2D,
                                        layerPaths: LayerPaths?
): Map<String, String> {
    val layerSeparableConv2DPaths = layerPaths as? LayerSeparableConv2DPaths
        ?: LayerSeparableConv2DPaths(
            layer.name,
            DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE,
            POINTWISE_KERNEL_DATA_PATH_TEMPLATE,
            DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
        )
    val variables = mutableMapOf(
        Pair(separableConv2dDepthwiseKernelVarName(layer.name), layerSeparableConv2DPaths.depthwiseKernelPath),
        Pair(separableConv2dPointwiseKernelVarName(layer.name), layerSeparableConv2DPaths.pointwiseKernelPath)
    )
    if (layer.useBias) {
        variables[separableConv2dBiasVarName(layer.name)] = layerSeparableConv2DPaths.biasPath
    }
    return variables
}

private fun getBatchNormVariables(layer: BatchNorm,
                                  layerPaths: LayerPaths?
): Map<String, String> {
    val layerBatchNormPaths = layerPaths as? LayerBatchNormPaths
        ?: LayerBatchNormPaths(
            layer.name,
            GAMMA_DATA_PATH_TEMPLATE,
            BETA_DATA_PATH_TEMPLATE,
            MOVING_MEAN_DATA_PATH_TEMPLATE,
            MOVING_VARIANCE_DATA_PATH_TEMPLATE
        )
    val variables = mutableMapOf(
        Pair(batchNormMovingMeanVarName(layer.name), layerBatchNormPaths.movingMeanPath),
        Pair(batchNormMovingVarianceVarName(layer.name), layerBatchNormPaths.movingVariancePath)
    )
    if (layer.scale) {
        variables[batchNormGammaVarName(layer.name)] = layerBatchNormPaths.gammaPath
    }
    if (layer.center) {
        variables[batchNormBetaVarName(layer.name)] = layerBatchNormPaths.betaPath
    }
    return variables
}

private fun getDenseVariables(layer: Dense,
                              layerPaths: LayerPaths?
): Map<String, String> {
    val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
        ?: LayerConvOrDensePaths(layer.name, KERNEL_DATA_PATH_TEMPLATE, BIAS_DATA_PATH_TEMPLATE)
    val variables = mutableMapOf(
        Pair(denseKernelVarName(layer.name), layerConvOrDensePaths.kernelPath)
    )
    if (layer.useBias) {
        variables[denseBiasVarName(layer.name)] = layerConvOrDensePaths.biasPath
    }
    return variables
}

/**
 * Parent class for specific paths to layers in h5 file. Contains only [layerName] field */
public open class LayerPaths(
    /** */
    public val layerName: String)

/**
 * Contains [layerName], [kernelPath], [biasPath] for specific layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerConvOrDensePaths(
    layerName: String,
    /** */
    public val kernelPath: String,
    /** */
    public val biasPath: String
) : LayerPaths(layerName)

/**
 * Contains [layerName], [depthwiseKernelPath],  [pointwiseKernelPath], [biasPath] for [SeparableConv2D] layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerSeparableConv2DPaths(
    layerName: String,
    /** */
    public val depthwiseKernelPath: String,
    /** */
    public val pointwiseKernelPath: String,
    /** */
    public val biasPath: String
) : LayerPaths(layerName)

/**
 * Contains [layerName], [gammaPath],  [betaPath], [movingMeanPath], [movingVariancePath] for [BatchNorm] layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerBatchNormPaths(
    layerName: String,
    /** */
    public val gammaPath: String,
    /** */
    public val betaPath: String,
    /** */
    public val movingMeanPath: String,
    /** */
    public val movingVariancePath: String
) : LayerPaths(layerName)
