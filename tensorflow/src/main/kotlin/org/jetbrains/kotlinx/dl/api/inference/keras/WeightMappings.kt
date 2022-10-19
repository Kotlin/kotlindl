/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import org.jetbrains.kotlinx.dl.api.core.layer.KVariable
import org.jetbrains.kotlinx.dl.api.core.layer.ParametrizedLayer
import org.jetbrains.kotlinx.dl.api.core.layer.activation.PReLU
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.AbstractConv
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvTranspose
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.DepthwiseConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.SeparableConv2D
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.normalization.BatchNorm
import org.jetbrains.kotlinx.dl.api.core.util.mapOfNotNull

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
    private const val PRELU_ALPHA_DATA_PATH_TEMPLATE = "/%s/%s/alpha:0"

    internal fun getLayerVariables(layer: ParametrizedLayer): Map<String, KVariable>? {
        return when (layer) {
            is Dense -> getDenseVariables(layer)
            is ConvTranspose -> getConvTransposeVariables(layer)
            is DepthwiseConv2D -> getDepthwiseConv2DVariables(layer)
            is SeparableConv2D -> getSeparableConv2DVariables(layer)
            is AbstractConv -> getConvVariables(layer)
            is BatchNorm -> getBatchNormVariables(layer)
            is PReLU -> getPReLUVariables(layer)
            else -> null
        }
    }

    internal fun getLayerVariablePathTemplates(
        layer: ParametrizedLayer,
        layerPaths: LayerPaths?
    ): Map<KVariable, String>? {
        return when (layer) {
            is Dense -> getDenseVariablesPathTemplates(layer, layerPaths)
            is ConvTranspose -> getConvTransposeVariablePathTemplates(layer, layerPaths)
            is DepthwiseConv2D -> getDepthwiseConv2DVariablePathTemplates(layer, layerPaths)
            is SeparableConv2D -> getSeparableConv2DVariablePathTemplates(layer, layerPaths)
            is AbstractConv -> getConvVariablePathTemplates(layer, layerPaths)
            is BatchNorm -> getBatchNormVariablePathTemplates(layer, layerPaths)
            is PReLU -> getPReLUVariablePathTemplates(layer, layerPaths)
            else -> null
        }
    }

    private fun getConvVariables(layer: AbstractConv): Map<String, KVariable> {
        return mapOfNotNull("kernel:0" to layer.kernel, "bias:0" to layer.bias)
    }

    private fun getConvVariablePathTemplates(layer: AbstractConv, layerPaths: LayerPaths?): Map<KVariable, String> {
        val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
            ?: LayerConvOrDensePaths(layer.name, KERNEL_DATA_PATH_TEMPLATE, BIAS_DATA_PATH_TEMPLATE)
        return mapOfNotNull(
            layer.kernel to layerConvOrDensePaths.kernelPath,
            layer.bias to layerConvOrDensePaths.biasPath
        )
    }

    private fun getConvTransposeVariables(layer: ConvTranspose): Map<String, KVariable> {
        return mapOfNotNull("kernel:0" to layer.kernel, "bias:0" to layer.bias)
    }

    private fun getConvTransposeVariablePathTemplates(
        layer: ConvTranspose,
        layerPaths: LayerPaths?
    ): Map<KVariable, String> {
        val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
            ?: LayerConvOrDensePaths(layer.name, KERNEL_DATA_PATH_TEMPLATE, BIAS_DATA_PATH_TEMPLATE)
        return mapOfNotNull(
            layer.kernel to layerConvOrDensePaths.kernelPath,
            layer.bias to layerConvOrDensePaths.biasPath
        )
    }

    private fun getDepthwiseConv2DVariables(layer: DepthwiseConv2D): Map<String, KVariable> {
        return mapOfNotNull("depthwise_kernel:0" to layer.kernel, "depthwise_bias:0" to layer.bias)
    }

    private fun getDepthwiseConv2DVariablePathTemplates(
        layer: DepthwiseConv2D,
        layerPaths: LayerPaths?
    ): Map<KVariable, String> {
        val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
            ?: LayerConvOrDensePaths(
                layer.name,
                DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE,
                DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
            )
        return mapOfNotNull(
            layer.kernel to layerConvOrDensePaths.kernelPath,
            layer.bias to layerConvOrDensePaths.biasPath
        )
    }

    private fun getSeparableConv2DVariables(layer: SeparableConv2D): Map<String, KVariable> {
        return mapOfNotNull(
            "depthwise_kernel:0" to layer.depthwiseKernel,
            "pointwise_kernel:0" to layer.pointwiseKernel,
            "bias:0" to layer.bias
        )
    }

    private fun getSeparableConv2DVariablePathTemplates(
        layer: SeparableConv2D,
        layerPaths: LayerPaths?
    ): Map<KVariable, String> {
        val layerSeparableConv2DPaths = layerPaths as? LayerSeparableConv2DPaths
            ?: LayerSeparableConv2DPaths(
                layer.name,
                DEPTHWISE_KERNEL_DATA_PATH_TEMPLATE,
                POINTWISE_KERNEL_DATA_PATH_TEMPLATE,
                DEPTHWISE_BIAS_DATA_PATH_TEMPLATE
            )
        return mapOfNotNull(
            layer.depthwiseKernel to layerSeparableConv2DPaths.depthwiseKernelPath,
            layer.pointwiseKernel to layerSeparableConv2DPaths.pointwiseKernelPath,
            layer.bias to layerSeparableConv2DPaths.biasPath
        )
    }

    private fun getDenseVariables(layer: Dense): Map<String, KVariable> {
        return mapOfNotNull("kernel:0" to layer.kernel, "bias:0" to layer.bias)
    }

    private fun getDenseVariablesPathTemplates(layer: Dense, layerPaths: LayerPaths?): Map<KVariable, String> {
        val layerConvOrDensePaths = layerPaths as? LayerConvOrDensePaths
            ?: LayerConvOrDensePaths(layer.name, KERNEL_DATA_PATH_TEMPLATE, BIAS_DATA_PATH_TEMPLATE)
        return mapOfNotNull(
            layer.kernel to layerConvOrDensePaths.kernelPath,
            layer.bias to layerConvOrDensePaths.biasPath
        )
    }

    private fun getBatchNormVariables(layer: BatchNorm): Map<String, KVariable> {
        return mapOfNotNull(
            "moving_mean:0" to layer.movingMean,
            "moving_variance:0" to layer.movingVariance,
            "gamma:0" to layer.gamma,
            "beta:0" to layer.beta
        )
    }


    private fun getBatchNormVariablePathTemplates(layer: BatchNorm, layerPaths: LayerPaths?): Map<KVariable, String> {
        val layerBatchNormPaths = layerPaths as? LayerBatchNormPaths
            ?: LayerBatchNormPaths(
                layer.name,
                GAMMA_DATA_PATH_TEMPLATE,
                BETA_DATA_PATH_TEMPLATE,
                MOVING_MEAN_DATA_PATH_TEMPLATE,
                MOVING_VARIANCE_DATA_PATH_TEMPLATE
            )
        return mapOfNotNull(
            layer.movingMean to layerBatchNormPaths.movingMeanPath,
            layer.movingVariance to layerBatchNormPaths.movingVariancePath,
            layer.gamma to layerBatchNormPaths.gammaPath,
            layer.beta to layerBatchNormPaths.betaPath
        )
    }

    private fun getPReLUVariables(layer: PReLU): Map<String, KVariable> {
        return mapOfNotNull("alpha:0" to layer.alpha)
    }

    private fun getPReLUVariablePathTemplates(layer: PReLU, layerPaths: LayerPaths?): Map<KVariable, String> {
        val layerPReLUPaths = layerPaths as? LayerPReLUPaths
            ?: LayerPReLUPaths(layer.name, PRELU_ALPHA_DATA_PATH_TEMPLATE)
        return mapOfNotNull(layer.alpha to layerPReLUPaths.alphaPath)
    }
}

/**
 * Parent class for specific paths to layers in h5 file. Contains only [layerName] field */
public open class LayerPaths(
    /** Name of the target layer. */
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
    /** Path to the kernel data. */
    public val kernelPath: String,
    /** Path to the bias data. */
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
    /** Path to depthwise kernel data. */
    public val depthwiseKernelPath: String,
    /** Path to pointwise kernel data.*/
    public val pointwiseKernelPath: String,
    /** Path to the bias data. */
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
    /** Path to gamma data. */
    public val gammaPath: String,
    /** Path to beta data. */
    public val betaPath: String,
    /** Path to moving mean data. */
    public val movingMeanPath: String,
    /** Path to moving variance data. */
    public val movingVariancePath: String
) : LayerPaths(layerName)

/**
 * Contains [layerName], [alphaPath] for [PReLU] layer, found in hdf5 file via
 * ```
 * recursivePrintGroupInHDF5File()
 * ```
 * function call.
 */
public class LayerPReLUPaths(
    layerName: String,
    /** Path to alpha data. */
    public val alphaPath: String
) : LayerPaths(layerName)
