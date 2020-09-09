package api.inference.keras

import api.conv2dBiasVarName
import api.conv2dKernelVarName
import api.denseBiasVarName
import api.denseKernelVarName
import api.keras.Sequential
import api.keras.layers.Dense
import api.keras.layers.Layer
import api.keras.layers.twodim.Conv2D
import io.jhdf.HdfFile

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
    model.addInitOpsToGraph(kernelVariableName, kernelData)
    model.addInitOpsToGraph(biasVariableName, biasData)
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

    model.addInitOpsToGraph(kernelVariableName, kernelData)
    model.addInitOpsToGraph(biasVariableName, biasData)
}


