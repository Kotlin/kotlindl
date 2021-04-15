/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.saveModelConfiguration
import org.tensorflow.Operand
import java.io.File
import java.io.FileNotFoundException

/**
 * A Functional model is defined as a directed graph of layers.
 *
 * @property [inputLayer] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Functional model via sequence of [layers].
 */
public class Functional(vararg layers: Layer) : GraphTrainableModel(*layers) {
    public companion object {
        /**
         * Creates the [Functional] model.
         *
         * @param [layers] The layers to describe the model design.
         * All connections between the layers must be established and form an acyclic directed graph.
         * Layers could be ordered in free way.
         *
         * NOTE: First layer should be input layer.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer): Functional {
            layerValidation(layers.toList())

            return preprocessAndCreate(layers.toList())
        }


        /**
         * Creates the [Functional] model.
         *
         * @param [layers] The layers to describe the model design.
         * All connections between the layers must be established and form an acyclic directed graph.
         * Layers could be ordered in free way.
         *
         * NOTE: First layer should be input layer.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>): Functional {
            layerValidation(layers)

            return preprocessAndCreate(layers)
        }

        /**
         * Creates the [Functional] model.
         *
         * @param [finalLayer] This layer specifies the output tensors that represent the outputs of this model.
         * All connections between the layers must be established and form an acyclic directed graph.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun fromOutput(finalLayer: Layer): Functional {
            require(finalLayer.inboundLayers.isNotEmpty()) { "Model should contain more than 1 layer!" }
            val layers = mutableSetOf<Layer>() // set of unique layers

            layers.add(finalLayer)
            visitInboundNodes(finalLayer, layers)

            return preprocessAndCreate(layers.toList())
        }

        private fun visitInboundNodes(finalLayer: Layer, layers: MutableSet<Layer>) {
            for (inboundNode in finalLayer.inboundLayers) {
                if (!layers.contains(inboundNode)) {
                    layers.add(inboundNode)
                    visitInboundNodes(inboundNode, layers)
                }
            }
        }

        private fun findInputLayer(layers: List<Layer>): Input {
            val inputs = layers.filterIsInstance<Input>().toList()
            require(inputs.size == 1) { "Model should contain only one layer with type Input. There is a ${inputs.size} input layers." }
            return inputs[0]
        }

        private fun fillOutputLayers(layers: List<Layer>) {
            layers.forEach { layer ->
                val inboundLayers = layer.inboundLayers
                inboundLayers.forEach {
                    if (!it.outboundLayers.contains(layer))
                        it.outboundLayers.add(layer)
                }
            }
        }

        private fun topologicalSort(layers: List<Layer>, inputLayer: Input): List<Layer> {
            val visited = mutableMapOf<Layer, Boolean>()
            layers.forEach { visited[it] = false }

            val grayStack: Stack<Layer> = mutableListOf()

            recursiveTopologicalSort(inputLayer, grayStack, visited)

            val sortedListOfLayers = mutableListOf<Layer>()
            while (grayStack.isNotEmpty())
                sortedListOfLayers.add(grayStack.pop()!!)

            return sortedListOfLayers
        }

        // Recursive topological Sort
        private fun recursiveTopologicalSort(node: Layer, stack: Stack<Layer>, visited: MutableMap<Layer, Boolean>) {
            val outboundLayers = node.outboundLayers
            for (i in 0 until outboundLayers.size) {
                val layer = outboundLayers[i]
                if (!visited[layer]!!) {
                    recursiveTopologicalSort(layer, stack, visited)
                    visited[layer] = true;
                }
            }
            stack.push(node)
        }

        /**
         * Creates the [Functional] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [Functional] model.
         */
        private fun preprocessAndCreate(layers: List<Layer>): Functional {
            var layerList = layers
            val inputLayer = findInputLayer(layerList)

            fillOutputLayers(layerList)
            layerList = topologicalSort(layerList, inputLayer)

            preProcessLayerNames(layerList.toTypedArray())
            val model = Functional(*layerList.toTypedArray())
            return model
        }

        /**
         * Loads a [Functional] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return Non-compiled and non-trained Functional model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File): Functional {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return List of layers. All connections between the layers are established and form an acyclic directed graph.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(configuration: File): MutableList<Layer> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return loadModelLayersFromConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Functional model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File): Functional {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadFunctionalModelConfiguration(configuration)
        }

        /**
         * Loads a [Functional] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return List of layers. All connections between the layers are established and form an acyclic directed graph.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(modelDirectory: File): MutableList<Layer> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadFunctionalModelLayers(configuration)
        }
    }

    override fun buildLayers() {
        inputLayer.build(tf)
        inputLayer.computeOutputShape()

        layers.filter { it !is Input }.forEach {
            it.buildFromInboundLayers(tf, kGraph)

            val outputShape = it.computeOutputShapeFromInboundLayers()
            val dims = outputShape.dims()

            check(outputShape.tail().all { elem -> elem > 0 })
            {
                "The last dimensions (except first = -1) of shape of layer ${it.name} contains zero or negative dimension values: ${dims.contentToString()}.\n" +
                        "Analyze your model architecture and layer output shapes carefully to discover a problem."
            }

            it.outputShape = outputShape //TODO: Refactoring: it could be done inside computeOutputShapeMethods

            logger.info { "${it.name}; outputShape: $outputShape $it" }
        }
    }

    override fun forward(input: Operand<Float>, inputLayer: Input): Operand<Float> {
        var output: Operand<Float> = input
        val outputByLayerName = mutableMapOf<String, Operand<Float>>()
        val outputs = mutableListOf<Operand<Float>>()
        outputs.add(input)
        outputByLayerName[inputLayer.name] = input
        for (layer in layers) {
            for (inboundLayer in layer.inboundLayers) {
                outputs.add(outputByLayerName[inboundLayer.name]!!)
            }
            output = layer.forward(tf, outputs, training, numberOfLossesOp)
            outputByLayerName[layer.name] = output
            outputs.clear()
        }
        return output
    }

    // TODO: check do we need a separate saving and cover by tests
    override fun save(
        modelDirectory: File,
        savingFormat: SavingFormat,
        saveOptimizerState: Boolean,
        writingMode: WritingMode
    ) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }
        //TODO: work wrong for cases in resnet50_prediction_save_load
        // check(isOptimizerVariableInitialized) { "The optimizer variables are not initialized yet. Initialize the optimizer variables with init() method or load optimizer weights to use this method." }

        val pathToModelDirectory = modelDirectory.absolutePath
        when (writingMode) {
            WritingMode.FAIL_IF_EXISTS -> {
                check(!modelDirectory.exists()) { "The directory exists on path $pathToModelDirectory, please be careful it could contain valuable model! Change this mode to OVERRIDE if you want to override this directory." }
                modelDirectory.mkdir()
            }
            WritingMode.OVERRIDE -> {
                if (modelDirectory.exists()) {
                    modelDirectory.deleteRecursively()
                }
                modelDirectory.mkdir()
            }
            WritingMode.APPEND -> {
                if (!modelDirectory.exists()) {
                    modelDirectory.mkdir()
                }
            }
        }

        when (savingFormat) {
            SavingFormat.TF_GRAPH_CUSTOM_VARIABLES -> saveInSimpleFormat(pathToModelDirectory, saveOptimizerState)
            SavingFormat.TF_GRAPH -> saveInSavedModelFormat(pathToModelDirectory)
            SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES -> saveInKerasFormat(pathToModelDirectory, saveOptimizerState)
        }
    }

    private fun saveInKerasFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveModel(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveModel(pathToModelDirectory: String, isKerasFullyCompatible: Boolean = true) {
        val jsonConfig = File("$pathToModelDirectory/modelConfig.json")
        this.saveModelConfiguration(
            jsonConfig,
            isKerasFullyCompatible = isKerasFullyCompatible
        )
    }

    private fun saveInSavedModelFormat(pathToModelDirectory: String) {
        saveGraphDef(pathToModelDirectory)
    }

    private fun saveInSimpleFormat(pathToModelDirectory: String, saveOptimizerState: Boolean) {
        saveGraphDef(pathToModelDirectory)
        saveVariables(pathToModelDirectory, saveOptimizerState)
    }

    private fun saveGraphDef(pathToModelDirectory: String) {
        val file = File("$pathToModelDirectory/graph.pb")
        file.writeBytes(kGraph.tfGraph.toGraphDef())
    }

    public override fun summary(
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int,
        stringParamSize: Int
    ): List<String> {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.info("==========================================================================================================")
        logger.info("Model: Functional")
        logger.info("__________________________________________________________________________________________________________")
        logger.info("Layer (type)                           Output Shape              Param #       Connected to               ")
        logger.info("==========================================================================================================")

        var totalTrainableParams = 0
        var totalFrozenParams = 0

        val layerDescriptions = mutableListOf<String>()

        for (l in layers) {
            if (l.isTrainable) totalTrainableParams += l.paramCount else totalFrozenParams += l.paramCount
            val inboundLayerNames = l.inboundLayers.map { it.name }.toTypedArray()

            if (inboundLayerNames.isNotEmpty()) {
                val layerDescription = createHeaderFunctionalLayerDescription(
                    l,
                    inboundLayerNames[0],
                    stringLayerNameTypeSize,
                    stringOutputShapeSize,
                    stringParamSize
                )
                layerDescriptions.add(layerDescription)
                logger.info(layerDescription)

                inboundLayerNames.drop(1).forEach {
                    val yetOneRowForInboundNode = createNextRowInFunctionalLayerDescription(
                        it,
                        stringLayerNameTypeSize + stringOutputShapeSize + stringParamSize
                    )
                    layerDescriptions.add(yetOneRowForInboundNode)
                    logger.info(yetOneRowForInboundNode)
                }

            } else {
                val layerDescription = createSimpleLayerDescription(l, stringLayerNameTypeSize, stringOutputShapeSize)
                layerDescriptions.add(layerDescription)
                logger.info(layerDescription)
            }

            logger.info("__________________________________________________________________________________________________________")
        }

        logger.info("==========================================================================================================")
        logger.info("Total trainable params: $totalTrainableParams")
        logger.info("Total frozen params: $totalFrozenParams")
        logger.info("Total params: ${totalTrainableParams + totalFrozenParams}")
        logger.info("==========================================================================================================")

        return layerDescriptions
    }

    private fun createSimpleLayerDescription(
        l: Layer,
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int
    ): String {
        val firstPart = "${l.name}(${l::class.simpleName})"

        val stringBuilder = StringBuilder(firstPart)
        for (i in 1 until stringLayerNameTypeSize - firstPart.length) {
            stringBuilder.append(" ")
        }

        val secondPart = l.outputShape.toString()

        stringBuilder.append(secondPart)

        for (i in 0 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(l.paramCount)

        return stringBuilder.toString()
    }

    private fun createHeaderFunctionalLayerDescription(
        l: Layer,
        inboundNodeName: String,
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int,
        stringParamSize: Int
    ): String {
        val firstPart = "${l.name}(${l::class.simpleName})"

        val stringBuilder = StringBuilder(firstPart)
        for (i in 1 until stringLayerNameTypeSize - firstPart.length) {
            stringBuilder.append(" ")
        }

        val secondPart = l.outputShape.toString()

        stringBuilder.append(secondPart)

        for (i in 0 until stringOutputShapeSize - secondPart.length) {
            stringBuilder.append(" ")
        }

        val thirdPart = l.paramCount.toString()

        stringBuilder.append(thirdPart)

        for (i in 0 until stringParamSize - thirdPart.length) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(inboundNodeName)

        return stringBuilder.toString()
    }

    private fun createNextRowInFunctionalLayerDescription(
        inboundNodeName: String,
        stringTabSize: Int
    ): String {
        val firstPart = ""

        val stringBuilder = StringBuilder(firstPart)
        for (i in 1 until stringTabSize) {
            stringBuilder.append(" ")
        }

        stringBuilder.append(inboundNodeName)

        return stringBuilder.toString()
    }
}
