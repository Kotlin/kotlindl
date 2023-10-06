/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.GpuConfiguration
import org.jetbrains.kotlinx.dl.api.core.floats
import org.jetbrains.kotlinx.dl.api.core.shape
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.tensorflow.Graph
import org.tensorflow.Session
import org.tensorflow.Tensor
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.NotDirectoryException

/**
 * Basic class for model inference.
 *
 * Provides functionality to make predictions and model loading.
 *
 * @property [tfGraph] TensorFlow computational graph.
 * @property [session] TensorFlow session.
 */
public open class TensorFlowInferenceModel(
    tfGraph: Graph = Graph(),
    session: Session = Session(tfGraph),
    config: GpuConfiguration? = null
) : TensorFlowInferenceModelBase(tfGraph = tfGraph, session = session, gpuConfiguration = config) {

    /** Input operand. */
    protected var input: String = DATA_PLACEHOLDER

    /** Output operand. */
    protected var output: String = OUTPUT_ARG_MAX

    override val inputDimensions: LongArray
        get() = TODO("Not yet implemented")

    override fun <T> predict(inputData: FloatData, extractResult: (TensorResult) -> T): T {
        return predict(mapOf(input to inputData), listOf(output), extractResult)
    }

    public fun <T> predict(
        inputData: FloatData,
        inputTensorName: String = input,
        outputTensorName: String = output,
        extractResult: (TensorResult) -> T
    ): T {
        return predict(mapOf(inputTensorName to inputData), listOf(outputTensorName), extractResult)
    }

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @param [inputTensorName] The name of input tensor.
     * @param [outputTensorName] The name of output tensor.
     * @return Predicted class index.
     */
    public fun predict(inputData: FloatData, inputTensorName: String = input, outputTensorName: String = output): Int {
        return predict(inputData, inputTensorName, outputTensorName) { result ->
            result.getLongArray(0)[0].toInt()
        }
    }

    /**
     * Setter for the input name.
     */
    public fun input(inputName: String) {
        input = inputName
    }

    /**
     * Setter for the output name.
     */
    public fun output(outputName: String) {
        output = outputName
    }

    /** Forms the graph description in string format. */
    public fun graphToString(): String {
        return tfGraph.convertToString()
    }

    override fun copy(): TensorFlowInferenceModel {
        return copy(copiedModelName = null)
    }

    /** Returns a copy of this model. */
    public fun copy(copiedModelName: String? = null): TensorFlowInferenceModel {
        val model = TensorFlowInferenceModel(tfGraph.copy())
        model.input = input
        model.output = output
        if (copiedModelName != null) model.name = name
        copyVariablesToModel(model, tfGraph.variableNames())
        model.isModelInitialized = true
        return model
    }

    override fun toString(): String {
        return "InferenceModel(name=$name)"
    }

    public companion object {
        private val logger = KotlinLogging.logger {}

        internal fun FloatData.toTensor(): Tensor<Float> {
            val preparedData = serializeToBuffer(floats)
            return Tensor.create(longArrayOf(1L, *shape.dims()), preparedData)
        }

        /**
         * Loads tensorflow graphs and variable data (if required).
         * It loads graph from .pb file format and variable data from .txt files
         *
         * @param [modelDirectory] Path to directory with TensorFlow graph and variable data.
         * @param [loadOptimizerState] Loads optimizer internal variables data, if true.
         */
        public fun load(
            modelDirectory: File,
            loadOptimizerState: Boolean = false
        ): TensorFlowInferenceModel {
            val pathToModelDirectory = modelDirectory.absolutePath
            if (!modelDirectory.exists()) {
                throw NotDirectoryException(pathToModelDirectory)
            }

            val file = File("$pathToModelDirectory/graph.pb")
            if (!file.exists()) throw FileNotFoundException(
                "File 'graph.pb' is not found. This file must be in the model directory. " +
                        "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.TF_GRAPH."
            )

            logger.debug { "Model loading started." }

            val model = TensorFlowInferenceModel(deserializeGraph(file.readBytes()))
            model.loadVariablesFromTxt(pathToModelDirectory, loadOptimizerState)
            model.isModelInitialized = true

            logger.debug { "Model loading finished." }

            return model
        }
    }
}