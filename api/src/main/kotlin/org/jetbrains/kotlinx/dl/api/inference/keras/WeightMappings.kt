/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.DepthwiseConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.SeparableConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.util.*

internal object WeightMappings {

    internal const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
    internal const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"
    private const val GAMMA_DATA_PATH_TEMPLATE = "/%s/%s/gamma:0"
    private const val BETA_DATA_PATH_TEMPLATE = "/%s/%s/beta:0"
    private const val MOVING_MEAN_DATA_PATH_TEMPLATE = "/%s/%s/moving_mean:0"
    private const val MOVING_VARIANCE_DATA_PATH_TEMPLATE = "/%s/%s/moving_variance:0"
    private const val DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_kernel:0"
    private const val POINTWISE_KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/pointwise_kernel:0"
    private const val DEPTHWISE_BIAS_DATA_PATH_TEMPLATE = "/%s/%s/depthwise_bias:0"

    // TODO: add loading for all layers with weights from Keras like Conv1D and Conv3D
    internal fun getLayerVariables(layer: Layer): Map<String, Pair<String, LongArray>>? {
        return when (layer) {
            is Dense -> getDenseVariables(layer)
            is Conv2D -> getConv2DVariables(layer)
            is DepthwiseConv2D -> getDepthwiseConv2DVariables(layer)
            is SeparableConv2D -> getSeparableConv2DVariables(layer)
            is BatchNorm -> getBatchNormVariables(layer)
            else -> null
        }
    }

    internal fun getLayerVariableNames(layer: Layer): List<String>? {
        return getLayerVariables(layer)?.map { (_, variable) -> variable.first }
    }

    internal fun getLayerVariablePathTemplates(layer: Layer, layerPaths: LayerPaths?): Map<String, String>? {
        return when (layer) {
            is Dense -> getDenseVariablesPathTemplates(layer, layerPaths)
            is Conv2D -> getConv2DVariablePathTemplates(layer, layerPaths)
            is DepthwiseConv2D -> getDepthwiseConv2DVariablePathTemplates(layer, layerPaths)
            is SeparableConv2D -> getSeparableConv2DVariablePathTemplates(layer, layerPaths)
            is BatchNorm -> getBatchNormVariablePathTemplates(layer, layerPaths)
            else -> null
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

    private fun getConv2DVariablePathTemplates(layer: Conv2D, layerPaths: LayerPaths?): Map<String, String> {
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

    private fun getDepthwiseConv2DVariables(layer: DepthwiseConv2D): Map<String, Pair<String, LongArray>> {
        val variables = mutableMapOf(
            Pair("depthwise_kernel:0", Pair(depthwiseConv2dKernelVarName(layer.name), layer.kernelShapeArray))
        )
        if (layer.useBias) {
            variables["depthwise_bias:0"] = Pair(depthwiseConv2dBiasVarName(layer.name), layer.biasShapeArray!!)
        }
        return variables
    }

    private fun getDepthwiseConv2DVariablePathTemplates(layer: DepthwiseConv2D,
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

    private fun getSeparableConv2DVariables(layer: SeparableConv2D): Map<String, Pair<String, LongArray>> {
        val variables = mutableMapOf(
            Pair(
                "depthwise_kernel:0",
                Pair(separableConv2dDepthwiseKernelVarName(layer.name), layer.depthwiseShapeArray)
            ),
            Pair(
                "pointwise_kernel:0",
                Pair(separableConv2dPointwiseKernelVarName(layer.name), layer.pointwiseShapeArray)
            )
        )
        if (layer.useBias) {
            variables["bias:0"] = Pair(separableConv2dBiasVarName(layer.name), layer.biasShapeArray!!)
        }
        return variables
    }

    private fun getSeparableConv2DVariablePathTemplates(layer: SeparableConv2D,
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

    private fun getDenseVariables(layer: Dense): Map<String, Pair<String, LongArray>> {
        val variables = mutableMapOf(
            Pair("kernel:0", Pair(denseKernelVarName(layer.name), layer.kernelShapeArray))
        )
        if (layer.useBias) {
            variables["bias:0"] = Pair(denseBiasVarName(layer.name), layer.biasShapeArray!!)
        }
        return variables
    }

    private fun getDenseVariablesPathTemplates(layer: Dense, layerPaths: LayerPaths?): Map<String, String> {
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


    private fun getBatchNormVariablePathTemplates(layer: BatchNorm, layerPaths: LayerPaths?): Map<String, String> {
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
}

/**
 * Parent class for specific paths to layers in h5 file. Contains only [layerName] field */
public open class LayerPaths(
    /** */
    public val layerName: String
)

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