/*
 * Copyright 2020-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.impl.util.use
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
public open class TensorFlowInferenceModel(tfGraph: Graph = Graph(),
                                           session: Session = Session(tfGraph)
) : TensorFlowInferenceModelBase(tfGraph, session) {

    /** Input operand. */
    protected var input: String = DATA_PLACEHOLDER

    /** Output operand. */
    protected var output: String = OUTPUT_ARG_MAX

    /** Data shape for prediction. */
    public lateinit var shape: LongArray
        private set

    /** Is true when shape is initialized. */
    protected val isShapeInitialized: Boolean
        get() = ::shape.isInitialized

    override val inputDimensions: LongArray
        get() = TODO("Not yet implemented")

    /**
     * Generates output prediction for the input sample.
     *
     * @param [inputData] Unlabeled input data to define label.
     */
    override fun predict(inputData: FloatArray): Int {
        require(::shape.isInitialized) { "Model input shape is not defined. Call reshape() to set input shape." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method." }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner = session.runner()

            return runner.feed(DATA_PLACEHOLDER, it)
                .fetch(output)
                .run().use { tensors ->
                    tensors.first().copyTo(LongArray(1))[0].toInt()
                }
        }
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        require(::shape.isInitialized) { "Model input shape is not defined. Call reshape() to set input shape." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method." }

        val fetchTensorName = predictionTensorName.ifEmpty { OUTPUT_NAME }

        require(tfGraph.operation(fetchTensorName) != null) { "No such tensor output named [$fetchTensorName] in the TensorFlow graph!" }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner1 = session.runner()
            return runner1.feed(DATA_PLACEHOLDER, it)
                .fetch(fetchTensorName)
                .run().use { tensors ->
                    tensors.first().convertTensorToMultiDimArray()[0] as FloatArray
                }
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

    override fun reshape(vararg dims: Long) {
        this.shape = TensorShape(1, *dims).dims()
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
        model.shape = shape
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