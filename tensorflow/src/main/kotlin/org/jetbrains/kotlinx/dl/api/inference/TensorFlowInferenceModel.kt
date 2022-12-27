/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.contentToString
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
import org.jetbrains.kotlinx.dl.impl.util.use
import org.tensorflow.Session
import org.tensorflow.Shape
import org.tensorflow.Tensor
import org.tensorflow.op.Ops
import java.io.File
import java.io.FileNotFoundException
import java.nio.file.NotDirectoryException
import java.util.*

/**
 * Basic class for model inference.
 *
 * Provides functionality to make predictions and model loading.
 */
public open class TensorFlowInferenceModel : InferenceModel {
    /** The namespace wrapper for all TensorFlow graph operations. */
    protected lateinit var tf: Ops

    /** TensorFlow session. */
    internal lateinit var session: Session

    /** TensorFlow wrapped computational graph. */
    public lateinit var kGraph: KGraph
        protected set

    /** Input operand. */
    protected var input: Input = Input.PLACEHOLDER

    /** Output operand. */
    protected var output: Output = Output.ARGMAX

    /** Data shape for prediction. */
    public lateinit var shape: LongArray
        private set

    /** Is true when shape is initialized. */
    protected val isShapeInitialized: Boolean
        get() = ::shape.isInitialized

    /** Is true when model is initialized. */
    public var isModelInitialized: Boolean = false
        internal set

    /** Logger. */
    private val logger = KotlinLogging.logger {}

    override val inputDimensions: LongArray
        get() = TODO("Not yet implemented")

    /** Model name. */
    public var name: String? = null

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
                .fetch(output.tfName)
                .run().use { tensors ->
                    tensors.first().copyTo(LongArray(1))[0].toInt()
                }
        }
    }

    override fun predictSoftly(inputData: FloatArray, predictionTensorName: String): FloatArray {
        require(::shape.isInitialized) { "Model input shape is not defined. Call reshape() to set input shape." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method." }

        val fetchTensorName = predictionTensorName.ifEmpty { OUTPUT_NAME }

        require(kGraph.tfGraph.operation(fetchTensorName) != null) { "No such tensor output named [$fetchTensorName] in the TensorFlow graph!" }

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
     * Chain-like setter to set up [inputOp].
     */
    public fun input(inputOp: Input) {
        input = inputOp
    }

    /**
     * Chain-like setter to set up [outputOp].
     */
    public fun output(outputOp: Output) {
        output = outputOp
    }

    override fun reshape(vararg dims: Long) {
        this.shape = TensorShape(1, *dims).dims()
    }

    /** Forms the graph description in string format. */
    public fun graphToString(): String {
        return kGraph.toString()
    }


    /** Closes internal resources: session and kGraph. */
    override fun close() {
        if (::session.isInitialized) {
            session.close()
        }
        if (::kGraph.isInitialized) {
            kGraph.close()
        }
    }

    override fun copy(
        copiedModelName: String?,
        saveOptimizerState: Boolean, // TODO, check this case
        copyWeights: Boolean
    ): TensorFlowInferenceModel {
        val model = TensorFlowInferenceModel()
        model.kGraph = this.kGraph.copy()
        model.tf = Ops.create(model.kGraph.tfGraph)
        model.session = Session(model.kGraph.tfGraph)
        model.shape = shape
        model.input = input
        model.output = output
        if (copiedModelName != null) model.name = name
        // TODO: check that tensors are closed after usage
        if (copyWeights) {
            val modelWeightsExtractorRunner = session.runner()
            val variableNames = kGraph.variableNames()
            check(variableNames.isNotEmpty()) {
                "Found 0 variable names in TensorFlow graph $kGraph. " +
                        "If copied model has no weights, set flag `copyWeights` to `false`."
            }

            val variableNamesToCopy = variableNames.filter { variableName ->
                saveOptimizerState || !isOptimizerVariable(variableName)
            }
            variableNamesToCopy.forEach(modelWeightsExtractorRunner::fetch)
            val modelWeights = variableNamesToCopy.zip(modelWeightsExtractorRunner.run()).toMap()

            model.loadVariables(modelWeights.keys) { variableName, _ ->
                modelWeights[variableName]!!.use { it.convertTensorToMultiDimArray() }
            }
        }
        model.isModelInitialized = true
        return model
    }

    /**
     * Loads variable data for variable names in the provided collection using a provided function.
     * @param [variableNames] Variable names to load.
     * @param [getData] Function that returns variable data by variable name and shape.
     */
    protected open fun loadVariables(variableNames: Collection<String>, getData: (String, Shape) -> Any) {
        for (variableName in variableNames) {
            val variableOperation = kGraph.tfGraph.operation(variableName)
            check(variableOperation != null) { "Operation $variableName is not found in static graph." }
            val variableShape = variableOperation.output<Float>(0).shape()
            val data = getData(variableName, variableShape)
            assignVariable(variableName, variableShape, data)
        }
    }

    /** Check that the variable with the name [variableName] is an optimizer variable**/
    protected fun isOptimizerVariable(variableName: String): Boolean = variableName.startsWith("optimizer")

    /**
     * Loads variable data from .txt files.
     *
     * @param [pathToModelDirectory] Path to directory with TensorFlow graph and variable data.
     * @param [loadOptimizerState] Loads optimizer internal variables data, if true.
     */
    protected fun loadVariablesFromTxt(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        loadVariablesFromTxt(pathToModelDirectory) { variableName ->
            loadOptimizerState || !isOptimizerVariable(variableName)
        }
    }

    /**
     * Loads variable data from .txt files for variables matching the provided predicate.
     *
     * @param [pathToModelDirectory] Path to directory with TensorFlow graph and variable data.
     * @param [predicate] Predicate for matching variable names for loading.
     */
    protected fun loadVariablesFromTxt(pathToModelDirectory: String, predicate: (String) -> Boolean) {
        val variableNamesFile = File("$pathToModelDirectory/variableNames.txt")

        if (!variableNamesFile.exists()) throw FileNotFoundException(
            "File 'variableNames.txt' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        val variableNamesToLoad = variableNamesFile.readLines().filter(predicate)
        loadVariables(variableNamesToLoad) { variableName, variableShape ->
            val file = File("$pathToModelDirectory/$variableName.txt")
            if (!file.exists()) throw FileNotFoundException(
                "File '$variableName.txt' is not found. This file must be in the model directory." +
                        "It is generated when saving the model with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
            )
            Scanner(file.inputStream()).use { scanner ->
                scanner.useLocale(Locale.US)
                scanner.createFloatArray(variableShape)
            }
        }
    }

    /**
     * Assigns variable data from multidimensional array.
     *
     * @param [variableName] Name of variable to load state for.
     * @param [variableShape] Shape of the variable.
     * @param [data] Variable data.
     */
    protected fun assignVariable(variableName: String, variableShape: Shape, data: Any) {
        val initializerName = defaultInitializerOpName(variableName)
        val assignOpName = defaultAssignOpName(variableName)

        val initOp = kGraph.tfGraph.operation(initializerName)
        check(initOp != null) {
            "Operation $initializerName is not found in static graph.\n" +
                    "NOTE: Loading of Zeros, Ones, Constant initializers is not supported."
        }

        val assignOp = kGraph.tfGraph.operation(assignOpName)
        check(assignOp != null) { "Operation $assignOp is not found in static graph." }

        populateVariable(assignOpName, initializerName, data)

        logger.debug { "Loading the variable $variableName data" }
        logger.debug { "Variable dimensions are: ${variableShape.contentToString()}" }
        logger.debug { "Number of elements: ${variableShape.numElements()}" }
    }

    private fun populateVariable(assignOpName: String, initializerName: String, data: Any) {
        var tensorData = data
        if (data is Array<*> && data.isArrayOf<Float>()) {
            tensorData = (data as Array<Float>).toFloatArray()
        }

        Tensor.create(tensorData).use { tensor ->
            session.runner()
                .feed(initializerName, tensor)
                .addTarget(assignOpName)
                .run()
        }
    }

    private fun loadModelFromSimpleFormat(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        inferenceGraphInitialization(pathToModelDirectory)
        loadVariablesFromTxt(pathToModelDirectory, loadOptimizerState)
    }

    private fun inferenceGraphInitialization(pathToModelDirectory: String) {
        val file = File("$pathToModelDirectory/graph.pb")

        if (!file.exists()) throw FileNotFoundException(
            "File 'graph.pb' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.TF_GRAPH."
        )

        kGraph = KGraph(file.readBytes())
        tf = Ops.create(kGraph.tfGraph)
        session = Session(kGraph.tfGraph)
    }

    override fun toString(): String {
        return "InferenceModel(name=$name)"
    }

    public companion object {
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
            val model = TensorFlowInferenceModel()

            val pathToModelDirectory = modelDirectory.absolutePath
            if (!modelDirectory.exists()) {
                throw NotDirectoryException(pathToModelDirectory)
            } else {
                model.logger.debug { "The model loading is started." }
                model.loadModelFromSimpleFormat(pathToModelDirectory, loadOptimizerState)
                model.isModelInitialized = true
                model.logger.debug { "The model loading is finished." }
            }

            return model
        }
    }
}