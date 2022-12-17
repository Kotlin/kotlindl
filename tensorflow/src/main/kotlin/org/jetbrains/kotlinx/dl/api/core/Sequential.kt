/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.setOutputShape
import org.jetbrains.kotlinx.dl.api.core.layer.weights
import org.jetbrains.kotlinx.dl.api.inference.keras.*
import org.tensorflow.Operand
import org.tensorflow.op.core.Placeholder
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

    override fun buildLayers(
        training: Operand<Boolean>,
        numberOfLosses: Operand<Float>
    ): Pair<Placeholder<Float>, Operand<Float>> {
        val input = inputLayer.build(tf)
        inputLayer.setOutputShape(input.asOutput().shape())
        var output: Operand<Float> = input

        layers.filter { it !is Input }.forEach { layer ->
            output = layer.build(tf, output, training, numberOfLossesOp)
            layer.setOutputShape(output.asOutput().shape())
        }

        return input to output
    }

    /** Returns a copy of this model. */
    // TODO: implement the saving of optimizer state
    public fun copy(saveOptimizerState: Boolean = false, copyWeights: Boolean = true): Sequential {
        val serializedModel = serializeModel(true)
        val deserializedModel = deserializeSequentialModel(serializedModel)
        if (!copyWeights) {
            return deserializedModel
        } else {
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

    public companion object {
        /**
         * Creates the [Sequential] model.
         *
         * @property [noInput] If true it disables input layer check.
         * @property [layers] The layers to describe the model design.
         *
         * NOTE: First layer should be an input layer, if you want to compile model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer, noInput: Boolean = false): Sequential {
            if (!noInput) {
                layerValidation(layers.toList())
            }

            preProcessLayerNames(layers)
            return Sequential(*layers)
        }

        /**
         * Creates the [Functional] model.
         *
         * @property [noInput] If true it disables input layer check.
         * @property [layers] The layers to describe the model design.
         *
         * NOTE: First layer should be an input layer, if you want to compile model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>, noInput: Boolean = false): Sequential {
            if (!noInput) {
                layerValidation(layers.toList())
            }

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
        public fun loadModelConfiguration(configuration: File, inputShape: IntArray? = null): Sequential {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            return loadSequentialModelConfiguration(configuration, inputShape)
        }

        /**
         * Loads a [Sequential] model layers from json file with model configuration.
         *
         * @param [configuration] File in .json format, containing the [Sequential] model.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromConfiguration(
            configuration: File,
            inputShape: IntArray? = null
        ): Pair<Input, List<Layer>> {
            require(configuration.isFile) { "${configuration.absolutePath} is not a file. Should be a .json file with configuration." }

            val config = loadSerializedModel(configuration)
            return loadSequentialModelLayers(config, inputShape)
        }

        /**
         * Loads a [Sequential] model from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Non-compiled and non-trained Sequential model.
         */
        @JvmStatic
        public fun loadDefaultModelConfiguration(modelDirectory: File, inputShape: IntArray? = null): Sequential {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a '$MODEL_CONFIG_JSON' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/$MODEL_CONFIG_JSON")

            if (!configuration.exists()) throw FileNotFoundException(
                "File '$MODEL_CONFIG_JSON' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            return loadSequentialModelConfiguration(configuration, inputShape)
        }

        /**
         * Loads a [Sequential] model layers from json file with name 'modelConfig.json' with model configuration located in [modelDirectory].
         *
         * @param [modelDirectory] Directory, containing file 'modelConfig.json'.
         * @throws [FileNotFoundException] If 'modelConfig.json' file is not found.
         * @return Pair of <input layer; list of layers>.
         */
        @JvmStatic
        public fun loadModelLayersFromDefaultConfiguration(
            modelDirectory: File,
            inputShape: IntArray? = null
        ): Pair<Input, List<Layer>> {
            require(modelDirectory.isDirectory) { "${modelDirectory.absolutePath} is not a directory. Should be a directory with a '$MODEL_CONFIG_JSON' file with configuration." }

            val configuration = File("${modelDirectory.absolutePath}/$MODEL_CONFIG_JSON")

            if (!configuration.exists()) throw FileNotFoundException(
                "File '$MODEL_CONFIG_JSON' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )

            val config = loadSerializedModel(configuration)
            return loadSequentialModelLayers(config, inputShape)
        }
    }
}
