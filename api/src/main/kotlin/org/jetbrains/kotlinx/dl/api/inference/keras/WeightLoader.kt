/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras

import io.jhdf.HdfFile
import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.layer.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.conv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.conv2dKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.denseBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.denseKernelVarName

private const val KERNEL_DATA_PATH_TEMPLATE = "/%s/%s/kernel:0"
private const val BIAS_DATA_PATH_TEMPLATE = "/%s/%s/bias:0"

/**
 * Loads weights from hdf5 file created in Keras TensorFlow framework.
 *
 * @param [hdfFile] File in hdf5 file format containing weights of Sequential model.
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeights(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
    this.logger.debug { "Starting weights loading.." }
    this.layers.forEach {
        run {
            when (it::class) {
                Dense::class -> fillDenseVariables(it.name, hdfFile, this, kernelDataPathTemplate, biasDataPathTemplate)
                org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D::class -> fillConv2DVariables(
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
public fun Sequential.loadWeights(
    hdfFile: HdfFile,
    layerList: MutableList<Layer>,
    kernelDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
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
                    org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D::class -> fillConv2DVariables(
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
                    org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, this)
                    else -> println("No default initialization handled for ${it.name}")
                }
                this.logger.info { " Weights initialized for ${it.name}. ${it.getParams()} parameters are initialized. " }
            }
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
 * @param [kernelDataPathTemplate] Template path to kernel weights of the specific layer.
 * @param [biasDataPathTemplate] Template path to bias weights of the specific layer.
 */
public fun Sequential.loadWeightsForFrozenLayers(
    hdfFile: HdfFile,
    kernelDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.KERNEL_DATA_PATH_TEMPLATE,
    biasDataPathTemplate: String = org.jetbrains.kotlinx.dl.api.inference.keras.BIAS_DATA_PATH_TEMPLATE
) {
    check(this.isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
    check(!isModelInitialized) { "Model is initialized already!" }
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
                    org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D::class -> fillConv2DVariables(
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
                    org.jetbrains.kotlinx.dl.api.core.layer.twodim.Conv2D::class -> initConv2DVariablesByDefaultInitializer(it.name, this)
                    else -> println("No default initialization handled for ${it.name}")
                }
                this.logger.info { " Weights initialized for ${it.name}. ${it.getParams()} parameters are initialized. " }
            }
        }
    }
    this.logger.info { "Weights are loaded." }
    this.isModelInitialized = true
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