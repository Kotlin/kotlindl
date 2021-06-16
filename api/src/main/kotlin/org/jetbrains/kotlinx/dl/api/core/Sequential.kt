/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.inference.keras.deserializeSequentialModel
import org.jetbrains.kotlinx.dl.api.inference.keras.loadSequentialModelLayers
import org.jetbrains.kotlinx.dl.api.inference.keras.loadSerializedModel
import org.jetbrains.kotlinx.dl.api.inference.keras.serializeModel
import org.tensorflow.Operand
import org.tensorflow.Shape
import java.io.File
import java.io.FileNotFoundException

/**
 * Sequential model groups a linear stack of layers into a TensorFlow Model.
 * Also, it provides training and inference features on this model.
 *
 * @property [inputLayer] the input layer with initial shapes.
 * @property [layers] the layers to describe the model design.
 * @constructor Creates a Sequential group with [inputLayer] and [layers].
 */
public class Sequential(vararg layers: Layer) : GraphTrainableModel(*layers) {
    public companion object {
        /**
         * Creates the [Sequential] model.
         *
         * @property [input] The input layer with initial shapes.
         * @property [layers] The layers to describe the model design.
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer): Sequential {
            layerValidation(layers.toList())

            preProcessLayerNames(layers)
            return Sequential(*layers)
        }

        /**
         * Creates the [Functional] model.
         * @property [layers] The layers to describe the model design.
         * NOTE: First layer should be input layer.
         * @return the [Functional] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>): Sequential {
            layerValidation(layers.toList())

            preProcessLayerNames(layers.toTypedArray())
            return Sequential(*layers.toTypedArray())
        }

        /**
         * Loads a [Sequential] model from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Sequential] model.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadModelConfiguration(configuration: File): Sequential {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadSequentialModelConfiguration(configuration)
        }

        /**
         * Loads a [Sequential] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Sequential] model.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(configuration: File): Pair<Input, MutableList<Layer>> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            val config = loadSerializedModel(configuration)
            return loadSequentialModelLayers(config)
        }

        /**
         * Loads a [Sequential] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File): Sequential {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return org.jetbrains.kotlinx.dl.api.inference.keras.loadSequentialModelConfiguration(configuration)
        }

        /**
         * Loads a [Sequential] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(modelDirectory: File): Pair<Input, MutableList<Layer>> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a 'modelConfig.json' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/modelConfig.json")

            if (!configuration.exists()) throw FileNotFoundException(
                "File 'modelConfig.json' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            val config = loadSerializedModel(configuration)
            return loadSequentialModelLayers(config)
        }
    }

    override fun buildLayers() {
        inputLayer.build(tf)
        var inputShape: Shape = inputLayer.computeOutputShape()

        layers.filter { it !is Input }.forEach {
            it.build(tf, kGraph, inputShape)

            inputShape = it.computeOutputShape(inputShape)
            val tensorShape = TensorShape(inputShape)
            val dims = tensorShape.dims()

            check(tensorShape.tail().all { elem -> elem > 0 })
            {
                "The last dimensions (except first = -1) of shape of layer ${it.name} contains zero or negative dimension values: ${dims.contentToString()}.\n" +
                        "Analyze your model architecture and layer output shapes carefully to discover a problem."
            }

            it.outputShape = tensorShape //TODO: Refactoring: it could be done inside computeOutputShapeMethods

            logger.debug { "${it.name}; $it; outputShape: $tensorShape" }
        }
    }

    override fun forward(input: Operand<Float>, inputLayer: Input): Operand<Float> {
        var out: Operand<Float> = input
        for (layer in layers) {
            out = layer.forward(tf, out, training, numberOfLossesOp)
        }
        return out
    }

    public override fun summary(
        stringLayerNameTypeSize: Int,
        stringOutputShapeSize: Int,
        stringParamSize: Int
    ): List<String> {
        check(isModelCompiled) { "The model is not compiled yet. Compile the model to use this method." }

        logger.info("===========================================================================")
        logger.info("Model: Sequential")
        logger.info("___________________________________________________________________________")
        logger.info("Layer (type)                           Output Shape              Param #   ")
        logger.info("===========================================================================")

        var totalTrainableParams = 0
        var totalFrozenParams = 0

        val layerDescriptions = mutableListOf<String>()

        for (l in layers) {
            if (l.isTrainable) totalTrainableParams += l.paramCount else totalFrozenParams += l.paramCount
            val layerDescription = createLayerDescription(l, stringLayerNameTypeSize, stringOutputShapeSize)
            layerDescriptions.add(layerDescription)
            logger.info(layerDescription)
            logger.info("___________________________________________________________________________")
        }

        logger.info("===========================================================================")
        logger.info("Total trainable params: $totalTrainableParams")
        logger.info("Total frozen params: $totalFrozenParams")
        logger.info("Total params: ${totalTrainableParams + totalFrozenParams}")
        logger.info("===========================================================================")

        return layerDescriptions
    }

    private fun createLayerDescription(
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

    /** Returns a copy of this model. */
    public fun copy(saveOptimizerState: Boolean = false, copyWeights: Boolean = true): Sequential {
        val serializedModel = serializeModel(true)
        val deserializedModel = deserializeSequentialModel(serializedModel)
        if (!copyWeights) {
            return deserializedModel
        } else {
            deserializedModel.compile(
                optimizer = this.optimizer,
                loss = this.loss,
                metric = this.metric
            )

            deserializedModel.layers.forEach {
                it.weights = this.getLayer(it.name).weights
            }

            deserializedModel.isModelInitialized = true

            return deserializedModel
        }
    }
}
