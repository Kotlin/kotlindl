/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.setOutputShape
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
public class Sequential public constructor(vararg layers: Layer, gpuConfiguration: GpuConfiguration?) :
    GraphTrainableModel(*layers, gpuConfiguration = gpuConfiguration) {

    public constructor(vararg layers: Layer) : this(*layers, gpuConfiguration = null)

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

    override fun copy(): Sequential {
        return copy(copiedModelName = null, copyOptimizerState = false, copyWeights = true)
    }

    /**
     * Creates a copy of this model.
     *
     * @param [copiedModelName] a name for the copy
     * @param [copyOptimizerState] whether optimizer state needs to be copied
     * @param [copyWeights] whether model weights need to be copied
     * @return A copied inference model.
     */
    public fun copy(
        copiedModelName: String? = null,
        copyOptimizerState: Boolean = false,
        copyWeights: Boolean = true
    ): Sequential {
        val serializedModel = serializeModel(true)
        return deserializeSequentialModel(serializedModel).also { modelCopy ->
            if (copiedModelName != null) modelCopy.name = copiedModelName
            if (copyWeights) copyWeightsTo(modelCopy, copyOptimizerState)
        }
    }

    public companion object {
        /**
         * Creates the [Sequential] model.
         *
         * @param [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         * @param [gpuConfiguration] The configuration of a model passed to the Tensorflow Runtime.
         *
         * NOTE: The first layer should be an input layer if you want to compile a model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(
            vararg layers: Layer,
            noInput: Boolean = false,
            gpuConfiguration: GpuConfiguration? = null
        ): Sequential {
            if (!noInput) {
                layerValidation(layers.toList())
            }

            preProcessLayerNames(layers)
            return Sequential(*layers, gpuConfiguration = gpuConfiguration)
        }

        /**
         * Creates the [Sequential] model.
         *
         * @param [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         *
         * NOTE: The first layer should be an input layer if you want to compile a model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(vararg layers: Layer, noInput: Boolean = false): Sequential {
            return of(layers = layers, noInput = noInput, gpuConfiguration = null)
        }

        /**
         * Creates the [Functional] model.
         *
         * @param [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         * @param [gpuConfiguration] The configuration of a model passed to the Tensorflow Runtime.
         *
         * NOTE: The first layer should be an input layer if you want to compile a model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(
            layers: List<Layer>,
            noInput: Boolean = false,
            gpuConfiguration: GpuConfiguration? = null
        ): Sequential {
            if (!noInput) {
                layerValidation(layers.toList())
            }

            preProcessLayerNames(layers.toTypedArray())
            return Sequential(*layers.toTypedArray(), gpuConfiguration = gpuConfiguration)
        }

        /**
         * Creates the [Sequential] model.
         *
         * @param [noInput] If true it disables input layer check.
         * @param [layers] The layers to describe the model design.
         *
         * NOTE: The first layer should be an input layer if you want to compile a model.
         *
         * @return the [Sequential] model.
         */
        @JvmStatic
        public fun of(layers: List<Layer>, noInput: Boolean = false): Sequential {
            return of(layers = layers, noInput = noInput, gpuConfiguration = null)
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
