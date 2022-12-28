/*
 * Copyright 2022-2023 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference

import mu.KotlinLogging
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.core.shape.contentToString
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.jetbrains.kotlinx.dl.api.core.util.createFloatArray
import org.jetbrains.kotlinx.dl.api.core.util.defaultAssignOpName
import org.jetbrains.kotlinx.dl.api.core.util.defaultInitializerOpName
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.jetbrains.kotlinx.dl.api.inference.TensorFlowInferenceModel.Companion.toTensor
import org.jetbrains.kotlinx.dl.impl.util.use
import org.tensorflow.*
import java.io.File
import java.io.FileNotFoundException
import java.util.*


/**
 * Base class for TensorFlow models.
 *
 * @property [tfGraph] TensorFlow computational graph.
 * @property [session] TensorFlow session.
 */
public abstract class TensorFlowInferenceModelBase(protected val tfGraph: Graph = Graph(),
                                                   internal val session: Session = Session(tfGraph)
) : InferenceModel<TensorResult> {

    /** Is true when model is initialized. */
    public var isModelInitialized: Boolean = false
        internal set

    /** Model name. */
    public var name: String? = null

    override fun <T> predict(inputs: Map<String, FloatData>,
                             outputs: List<String>,
                             extractResult: (TensorResult) -> T
    ): T {
        check(isModelInitialized) { "Model weights are not initialized." }
        outputs.forEach { outputName ->
            require(tfGraph.operation(outputName) != null) {
                "Could not find tensor output $outputName in the TensorFlow graph."
            }
        }

        return runModel(
            inputs.map { InputKey.Name(it.key) to it.value.toTensor() }.toMap(),
            outputs.map { OutputKey.Name(it) },
            emptyList(),
        ) { tensors -> extractResult(TensorResult(tensors)) }
    }

    protected fun <R> runModel(inputs: Map<out InputKey, Tensor<*>>,
                               outputs: List<OutputKey>,
                               targets: List<Operand<Float>>,
                               extractResult: (List<Tensor<*>>) -> R
    ): R {
        return inputs.use {
            val runner = session.runner()
            inputs.forEach { (operation, tensor) -> operation.feed(runner, tensor) }
            outputs.forEach { output -> output.fetch(runner) }
            targets.forEach { target -> runner.addTarget(target) }
            runner.run().use { tensors -> extractResult(tensors) }
        }
    }

    protected sealed class InputKey {
        public data class Operand(val op: org.tensorflow.Operand<*>) : InputKey() {
            override fun feed(runner: Session.Runner, tensor: Tensor<*>): Session.Runner {
                return runner.feed(op.asOutput(), tensor)
            }
        }

        public data class Name(val s: String) : InputKey() {
            override fun feed(runner: Session.Runner, tensor: Tensor<*>): Session.Runner = runner.feed(s, tensor)
        }

        public abstract fun feed(runner: Session.Runner, tensor: Tensor<*>): Session.Runner
    }

    protected sealed class OutputKey {
        public data class Operand(val op: org.tensorflow.Operand<Float>) : OutputKey() {
            override fun fetch(runner: Session.Runner): Session.Runner = runner.fetch(op)
        }

        public data class Name(val s: String) : OutputKey() {
            override fun fetch(runner: Session.Runner): Session.Runner = runner.fetch(s)
        }

        public abstract fun fetch(runner: Session.Runner): Session.Runner
    }

    /**
     * Loads variable data for variable names in the provided collection using a provided function.
     * @param [variableNames] Variable names to load.
     * @param [getData] Function that returns variable data by variable name and shape.
     */
    protected open fun loadVariables(variableNames: Collection<String>, getData: (String, Shape) -> Any) {
        for (variableName in variableNames) {
            val variableOperation = tfGraph.operation(variableName)
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

        val initOp = tfGraph.operation(initializerName)
        check(initOp != null) {
            "Operation $initializerName is not found in static graph.\n" +
                    "NOTE: Loading of Zeros, Ones, Constant initializers is not supported."
        }

        val assignOp = tfGraph.operation(assignOpName)
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

    protected fun copyVariablesToModel(model: TensorFlowInferenceModelBase, variableNames: List<String>) {
        if (variableNames.isEmpty()) return

        val modelWeightsExtractorRunner = session.runner()
        variableNames.forEach(modelWeightsExtractorRunner::fetch)
        val modelWeights = variableNames.zip(modelWeightsExtractorRunner.run()).toMap()

        model.loadVariables(modelWeights.keys) { variableName, _ ->
            modelWeights[variableName]!!.use { it.convertTensorToMultiDimArray() }
        }
    }

    /** Closes internal resources: session and tfGraph. */
    override fun close() {
        session.close()
        tfGraph.close()
    }

    private companion object {
        private val logger = KotlinLogging.logger {}
    }
}

/**
 * Inference result for the models inheriting [TensorFlowInferenceModelBase].
 */
public data class TensorResult(val tensors: List<Tensor<*>>) : AutoCloseable {
    override fun close() {
        tensors.forEach {
            try {
                it.close()
            } finally {
            }
        }
    }
}