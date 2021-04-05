/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.util.*
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Input
import org.jetbrains.kotlinx.dl.api.inference.savedmodel.Output
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
public open class InferenceModel : AutoCloseable {
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
    protected lateinit var shape: LongArray

    protected val isShapeInitialized: Boolean
        get() = ::shape.isInitialized

    /** Is true when model is initialized. */
    public var isModelInitialized: Boolean = false
        internal set

    /** Logger. */
    private val logger = KotlinLogging.logger {}

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
        ): InferenceModel {
            val model = InferenceModel()

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

    /**
     * Predicts the class of [inputData].
     *
     * @param [inputData] The single example with unknown label.
     * @return Predicted class index.
     */
    public open fun predict(inputData: FloatArray): Int {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method." }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner = session.runner()

            val result = runner.feed(DATA_PLACEHOLDER, it)
                .fetch(output.tfName)
                .run()[0]

            return result.copyTo(LongArray(1))[0].toInt()
        }

    }

    /**
     * Predicts vector of probabilities instead of specific class in [predict] method.
     *
     * @param [inputData] The single example with unknown vector of probabilities.
     * @param [predictionTensorName] The name of prediction tensor. It could be changed, if you need to get alternative outputs from intermediate parts of the TensorFlow graphs.
     * @return Vector that represents the probability distributions of a list of potential outcomes
     */
    public open fun predictSoftly(inputData: FloatArray, predictionTensorName: String = ""): FloatArray {
        require(::shape.isInitialized) { "Reshape functions is missed! Define and set up the reshape function to transform initial data to the model input." }
        check(isModelInitialized) { "The model is not initialized yet. Initialize the model weights with InferenceModel.load() method." }

        val fetchTensorName = if (predictionTensorName.isEmpty()) OUTPUT_NAME else predictionTensorName

        require(kGraph.tfGraph.operation(fetchTensorName) != null) { "No such tensor output named [$fetchTensorName] in the TensorFlow graph!" }

        val preparedData = serializeToBuffer(inputData)
        val tensor = Tensor.create(shape, preparedData)

        tensor.use {
            val runner1 = session.runner()
            val result1 = runner1.feed(DATA_PLACEHOLDER, it)
                .fetch(fetchTensorName)
                .run()[0]

            val arr = Array(1) { FloatArray(10) { 0.0f } }
            result1.copyTo(arr)

            return arr[0]
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

    /**
     * Chain-like setter to set up [reshapeFunction] function.
     *
     * @param reshapeFunction The approach to convert raw input data to multidimensional array of floats.
     */
    public fun reshape(vararg dims: Long) {
        this.shape = TensorShape(1, *dims).dims()
    }

    override fun toString(): String {
        return "Model contains $kGraph"
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

    /**
     * Executes pre-defined Assign TensorFlow operand.
     *
     * @param [variableName] Name of variable to be assigned.
     */
    internal fun runAssignOpByVarName(
        variableName: String
    ) {
        val assignOpName = defaultAssignOpName(variableName)

        session.runner()
            .addTarget(assignOpName)
            .run()
    }

    /**
     * Fills variable with the given data in appropriate form (shape).
     *
     * @param variableName Name of variable to be filled.
     * @param kernelData Data for variable filling, should have correct shape and type.
     */
    internal fun fillVariable(
        variableName: String,
        kernelData: Any
    ) {
        val initializerName = defaultInitializerOpName(variableName)
        val assignOpName = defaultAssignOpName(variableName)

        populateVariable(initializerName, kernelData, assignOpName)
    }

    /**
     * Loads variable data from .txt files.
     *
     * @param [pathToModelDirectory] Path to directory with TensorFlow graph and variable data.
     * @param [loadOptimizerState] Loads optimizer internal variables data, if true.
     */
    protected fun loadVariablesFromTxt(pathToModelDirectory: String, loadOptimizerState: Boolean) {
        val file = File("$pathToModelDirectory/variableNames.txt")

        if (!file.exists()) throw FileNotFoundException(
            "File 'variableNames.txt' is not found. This file must be in the model directory. " +
                    "It is generated during Sequential model saving with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        val variableNames = file.readLines()
        if (variableNames.isNotEmpty()) {
            for (variableName in variableNames) {
                if (!loadOptimizerState && variableName.startsWith("optimizer")) // skip loading optimizers' variables
                    continue
                loadVariable(variableName, pathToModelDirectory)
            }
        }
    }

    /**
     * Loads variable data from .txt file.
     *
     * @param [variableName] Name of variable to load state.
     * @param [pathToModelDirectory] Path to directory with TensorFlow graph and variable data.
     */
    protected fun loadVariable(variableName: String, pathToModelDirectory: String) {
        val operation = kGraph.tfGraph.operation(variableName)
        check(operation != null) { "Operation $variableName is not found in static graph." }

        val file = File("$pathToModelDirectory/$variableName.txt")

        if (!file.exists()) throw FileNotFoundException(
            "File '$variableName.txt' is not found. This file must be in the model directory." +
                    "It is generated when saving the model with SavingFormat.TF_GRAPH_CUSTOM_VARIABLES or SavingFormat.JSON_CONFIG_CUSTOM_VARIABLES."
        )

        Scanner(file.inputStream()).use { scanner ->
            scanner.useLocale(Locale.US)
            val initializerName = defaultInitializerOpName(variableName)
            val assignOpName = defaultAssignOpName(variableName)

            val initOp = kGraph.tfGraph.operation(initializerName)
            check(initOp != null) {
                "Operation $initializerName is not found in static graph.\n" +
                        "NOTE: Loading of Zeros, Ones, Constant initializers is not supported."
            }

            val assignOp = kGraph.tfGraph.operation(assignOpName)
            check(assignOp != null) { "Operation $assignOp is not found in static graph." }

            val shape = operation.output<Float>(0).shape()
            val tensorShape = TensorShape(shape)

            val source = createFloatArrayFromScanner(shape, scanner)
            populateVariable(initializerName, source, assignOpName)

            logger.debug { "Loading the variable $variableName data" }
            logger.debug { "Variable dimensions are: ${tensorShape.dims().contentToString()}" }
            logger.debug { "Amount of elements: ${tensorShape.numElements()}" }
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

    private fun createFloatArrayFromScanner(shape: Shape, scanner: Scanner): Any {
        when (shape.numDimensions()) {
            0 -> {
                return scanner.nextFloat()
            }
            1 -> {
                return create1DimFloatArray(shape, scanner)
            }
            2 -> {
                return create2DimFloatArray(shape, scanner)
            }
            3 -> {
                return create3DimFloatArray(shape, scanner)
            }
            4 -> {
                return create4DimFloatArray(shape, scanner)
            }
            else -> {
                throw RuntimeException("The loading of tensors with 5 and more dimensions is not supported yet")
            }
        }
    }

    private fun populateVariable(
        initializerName: String,
        data: Any,
        assignOpName: String
    ) {
        Tensor.create(data).use { tensor ->
            session.runner()
                .feed(initializerName, tensor)
                .addTarget(assignOpName)
                .run()

        }
    }

    private fun create4DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<Array<Array<FloatArray>>> {
        val result = Array(shape.size(0).toInt()) {
            Array(shape.size(1).toInt()) {
                Array(shape.size(2).toInt()) {
                    FloatArray(shape.size(3).toInt()) { 0.0f }
                }
            }
        }

        var cnt = 0

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    for (m in result[i][j][k].indices) {
                        if (scanner.hasNextFloat()) {
                            val weight = scanner.nextFloat()
                            result[i][j][k][m] = weight
                            cnt++
                        } else {
                            logger.debug { cnt }
                        }
                    }
                }
            }
        }

        return result
    }

    private fun create3DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<Array<FloatArray>> {
        val result = Array(shape.size(0).toInt()) {
            Array(shape.size(1).toInt()) {
                FloatArray(shape.size(2).toInt()) { 0.0f }
            }
        }

        for (i in result.indices) {
            for (j in result[i].indices) {
                for (k in result[i][j].indices) {
                    val weight = scanner.nextFloat()
                    result[i][j][k] = weight
                }
            }
        }

        return result
    }

    private fun create2DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): Array<FloatArray> {
        val result = Array(shape.size(0).toInt()) {
            FloatArray(shape.size(1).toInt()) { 0.0f }
        }

        for (i in result.indices) {
            for (j in result[i].indices) {
                val weight = scanner.nextFloat()
                result[i][j] = weight
            }
        }

        return result
    }

    private fun create1DimFloatArray(
        shape: Shape,
        scanner: Scanner
    ): FloatArray {
        val result = FloatArray(shape.size(0).toInt()) { 0.0f }

        for (i in result.indices) {
            val weight = scanner.nextFloat()
            result[i] = weight
        }

        return result
    }
}
