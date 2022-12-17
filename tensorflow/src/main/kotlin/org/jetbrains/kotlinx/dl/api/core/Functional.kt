/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.freeze
import org.jetbrains.kotlinx.dl.api.core.layer.setOutputShape
import org.jetbrains.kotlinx.dl.api.core.layer.weights
import org.jetbrains.kotlinx.dl.api.core.util.sortTopologically
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.tensorflow.Operand
import org.tensorflow.op.core.Placeholder
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

    override fun buildLayers(
        training: Operand<Boolean>,
        numberOfLosses: Operand<Float>
    ): Pair<Placeholder<Float>, Operand<Float>> {
        val input = inputLayer.build(tf)
        inputLayer.setOutputShape(input.asOutput().shape())
        val output = mutableMapOf<Layer, Operand<Float>>(inputLayer to input)

        layers.filter { it !is Input }.forEach { layer ->
            val out = layer.build(tf, layer.inboundLayers.map { output[it]!! }, training, numberOfLosses)
            output[layer] = out
            layer.setOutputShape(out.asOutput().shape())
        }

        return input to output[layers.last()]!!
    }

    override fun save(
        modelDirectory: File,
        savingFormat: SavingFormat,
        saveOptimizerState: Boolean,
        writingMode: WritingMode
    ) {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with init() method or load weights to use this method." }
        if (saveOptimizerState) {
            check(isOptimizerVariableInitialized) { "The optimizer variables are not initialized yet. Initialize the optimizer variables with init() method or load optimizer weights to use this method." }
        }

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
        val jsonConfig = File("$pathToModelDirectory/$MODEL_CONFIG_JSON")
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

    /** Returns a copy of this model. */
    // TODO: support saveOptimizerState=true with assignment of intermediate optimizer state
    public fun copy(saveOptimizerState: Boolean = false, copyWeights: Boolean = true): Functional {
        val serializedModel = serializeModel(true)
        val deserializedModel = deserializeFunctionalModel(serializedModel)
        if (!copyWeights) {
            return deserializedModel
        } else {
            // TODO: make deep copies, not just links
            deserializedModel.compile(
                optimizer = this.optimizer,
                loss = this.loss,
                metrics = this.metrics
            )

            deserializedModel.layers.forEach {
                it.weights = this.getLayer(it.name).weights
            }

            deserializedModel.isModelInitialized = true

            return deserializedModel
        }
    }

    /** Removes the last layer from the [Functional] model, if it's not compiled yet! . */
    public fun removeLastLayer(): Functional {
        require(!this.isModelCompiled) { "It works for non-compiled models only!" }

        val layers = mutableListOf<Layer>()

        for (layer in this.layers) {
            layers.add(layer)
        }

        val lastLayer = layers.last()
        for (outboundLayer in lastLayer.inboundLayers)
            outboundLayer.outboundLayers.remove(lastLayer)

        layers.removeLast()

        return of(layers)
    }

    public companion object {
        /**
         * Creates the [Functional] model.
         *
         * @property [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         * All connections between the layers must be established and form an acyclic directed graph.
         * Layers could be ordered in free way.
         *
         * NOTE: First layer should be an input layer, if you want to compile model.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer, noInput: Boolean = false): Functional {
            if (!noInput) {
                layerValidation(layers.toList())
            }

            return preprocessAndCreate(layers.toList())
        }

        /**
         * Creates the [Functional] model.
         *
         * @property [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         * All connections between the layers must be established and form an acyclic directed graph.
         * Layers could be ordered in free way.
         *
         * NOTE: First layer should be an input layer, if you want to compile model.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>, noInput: Boolean = false): Functional {
            if (!noInput) {
                layerValidation(layers.toList())
            }

            return preprocessAndCreate(layers)
        }

        /**
         * Creates the [Functional] model from two models: [pretrainedModel] and [topModel].
         * All layers of pretrainedModel will be frozen automatically.
         * The input of the [topModel] will be connected to the output of the [pretrainedModel].
         *
         * NOTE: First layer of [pretrainedModel] should be an input layer.
         * NOTE: Both models should be non-compiled yet.
         *
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(pretrainedModel: GraphTrainableModel, topModel: GraphTrainableModel): Functional {
            require(!pretrainedModel.isModelCompiled) { "Pretrained model should not be compiled!" }
            require(!topModel.isModelCompiled) { "Top model should not be compiled!" }

            // TODO: make a honest copy of models (pretrained and topModel) or of all layers
            val pretrainedLayers = pretrainedModel.layers
            // Freezing
            pretrainedLayers.forEach { it.freeze() }

            val layers = mutableListOf<Layer>()
            layers += pretrainedLayers

            val topLayers = topModel.layers
            layers += topLayers
            topLayers[0].inboundLayers.add(pretrainedLayers.last())

            if (topModel is Sequential && layers.size > 1) {
                // establish edges in DAG
                topLayers.subList(1, topLayers.size).forEachIndexed { index, layer ->
                    val topLayersIndex = index - 1 + 1
                    // shift -1 to take previous, but shift +1 because it's an index in subList, started from 1
                    layer.inboundLayers.add(topLayers[topLayersIndex])
                }
            }

            return of(layers)
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
            val sortedListOfLayers = sortTopologically(inputLayer, Layer::outboundLayers)
            check(sortedListOfLayers.size == layers.size) {
                "The following layers are not reachable from the input: ${
                    layers.minus(sortedListOfLayers.toSet()).map { it.name + " (" + it::class.simpleName + ")" }
                }"
            }
            return sortedListOfLayers
        }

        /**
         * Creates the [Functional] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [Functional] model.
         */
        private fun preprocessAndCreate(layers: List<Layer>): Functional {
            val inputLayer = findInputLayer(layers)

            fillOutputLayers(layers)
            val layerList = topologicalSort(layers, inputLayer)

            preProcessLayerNames(layerList.toTypedArray())
            return Functional(*layerList.toTypedArray())
        }

        /**
         * Loads a [Functional] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return Non-compiled and non-trained Functional model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File, inputShape: IntArray? = null): Functional {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return loadFunctionalModelConfiguration(configuration, inputShape)
        }

        /**
         * Loads a [Functional] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Functional] model.
         * @return List of layers. All connections between the layers are established and form an acyclic directed graph.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(
            configuration: File,
            inputShape: IntArray? = null
        ): List<Layer> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            val functionalConfig = loadSerializedModel(configuration)
            return loadFunctionalModelLayers(functionalConfig, inputShape)
        }

        /**
         * Loads a [Functional] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Functional model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File, inputShape: IntArray? = null): Functional {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a '$MODEL_CONFIG_JSON' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/$MODEL_CONFIG_JSON")

            if (!configuration.exists()) throw FileNotFoundException(
                "File '$MODEL_CONFIG_JSON' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadFunctionalModelConfiguration(configuration, inputShape)
        }

        /**
         * Loads a [Functional] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return List of layers. All connections between the layers are established and form an acyclic directed graph.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(
            modelDirectory: File,
            inputShape: IntArray? = null
        ): List<Layer> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a '$MODEL_CONFIG_JSON' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/$MODEL_CONFIG_JSON")

            if (!configuration.exists()) throw FileNotFoundException(
                "File '$MODEL_CONFIG_JSON' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            val functionalConfig = loadSerializedModel(configuration)
            return loadFunctionalModelLayers(functionalConfig, inputShape)
        }
    }
}
