/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.AssignAdd
import org.tensorflow.op.core.Variable

/**
 * A wrapper object to extend functionality and behaviour of static TensorFlow graph.
 *
 * It tracks all model variables (used in optimizers or layers) and its initializers.
 *
 * @param [graphDef] A serialized representation of the graph.
 * @param [prefix] A prefix that will be prepended to names in graphDef.
 *
 * @constructor Creates KGraph by serialized representation of the graph.
 */
public class KGraph(graphDef: ByteArray, prefix: String) : AutoCloseable {
    public constructor(graphDef: ByteArray) : this(graphDef, "")

    /** Internal static TensorFlow graph. */
    internal var tfGraph: Graph = Graph()

    /** True if the graph object is closed and the occupied resources are freed. */
    public var isClosed: Boolean = false

    /** A list of initializer to initialize the trainableVariables. */
    private val optimizerInitializers: MutableList<Assign<*>> = mutableListOf()

    /** A list of initializer to initialize the trainableVariables. */
    private val optimizerAssignAddInitializers: MutableList<AssignAdd<*>> = mutableListOf()

    /** A list of variables to train. */
    private val layerVariables: MutableMap<Variable<Float>, Boolean> = mutableMapOf()

    /** A list of variables to train. */
    internal val variableRegularizers: MutableMap<Variable<Float>, Regularizer> = mutableMapOf()

    /** A list of optimizers' variables. */
    private val optimizerVariables: MutableList<Variable<Float>> = mutableListOf()

    /** A list of initializer to initialize the trainableVariables. */
    private val initializers: MutableMap<String, Assign<Float>> = mutableMapOf()

    init {
        if (prefix.isEmpty()) {
            tfGraph.importGraphDef(graphDef)
        } else {
            tfGraph.importGraphDef(graphDef, prefix)
        }
    }

    /**
     * Closes internal TensorFlow graph.
     */
    override fun close() {
        isClosed = true
        tfGraph.close()
    }

    override fun toString(): String {
        return convertGraphDefToString()
    }

    private fun convertGraphDefToString(): String {
        val operations = tfGraph.operations()

        var s = ""
        while (operations.hasNext()) {
            val operation = operations.next() as GraphOperation
            s += "Name: " + operation.name() + "; Type: " + operation.type() + "; Out #tensors:  " + operation.numOutputs() + "\n"
        }
        return s
    }

    /** Returns list of variable names in TensorFlow graph. */
    public fun variableNames(): List<String> {
        val operations = tfGraph.operations()
        val variableNames = mutableListOf<String>()

        while (operations.hasNext()) {
            val operation = operations.next() as GraphOperation
            if(operation.type().equals("VariableV2")) {
                variableNames.add(operation.name())
            }
        }
        return variableNames.toList()
    }

    /** Makes a graph copy. */
    public fun copy(): KGraph {
        require(!isClosed) { "The copied graph and model are closed and could not be reused!" }
        return KGraph(tfGraph.toGraphDef())
    }

    /**
     * Adds a variable used in layer to the pool of tracked variables.
     *
     * @param variable Variable to track in KGraph.
     * @param isTrainable When passed isTrainable = true, the KGraph automatically adds new variables to the tracked collection of graph variables.
     */
    public fun addLayerVariable(variable: Variable<Float>, isTrainable: Boolean) {
        check(!layerVariables.contains(variable)) { "$variable is added to graph already. Analyze and fix the static graph building process." }
        layerVariables[variable] = isTrainable
    }

    /**
     * Adds a variable related to the given [regularizer] to the pool of tracked variables.
     *
     * @param [variable] TensorFlow variable to track in KGraph.
     * @param [regularizer] The regularizer related to the given [variable].
     */
    public fun addVariableRegularizer(variable: Variable<Float>, regularizer: Regularizer) {
        check(!variableRegularizers.contains(variable)) { "$variable is added to graph already. Analyze and fix the static graph building process." }
        variableRegularizers[variable] = regularizer
    }

    /**
     * Adds a variable used in optimizer to the pool of tracked variables.
     *
     * @param variable Optimizer variable to track in KGraph.
     */
    public fun addOptimizerVariable(variable: Variable<Float>) {
        check(!optimizerVariables.contains(variable)) { "$variable is added to graph already. Analyze and fix the static graph building process." }
        optimizerVariables.add(variable)
    }

    /**
     * Adds an initializer for layer variable tracked in KGraph.
     *
     * @param variableName Variable name to bind the variable and initializer.
     * @param initializer Assign TensorFlow operand to initialize variable with name: [variableName].
     */
    public fun addInitializer(variableName: String, initializer: Assign<Float>) {
        check(!initializers.contains(variableName)) { "$variableName has initializer already. Analyze and fix the static graph building process." }
        initializers[variableName] = initializer
    }

    /**
     * Adds an optimizer initializer for optimizer variable tracked in KGraph.
     *
     * @param initializer Assign TensorFlow operand to initialize optimizer variable.
     */
    public fun addOptimizerVariableInitializer(initializer: Assign<*>) {
        optimizerInitializers += initializer
    }

    /**
     * Adds an optimizer initializer of special 'AssignAdd' type for optimizer variable tracked in KGraph.
     *
     * @param initializer AssignAdd TensorFlow operand to initialize and increase optimizer variable.
     */
    public fun addOptimizerVariableAssignAddInitializer(initializer: AssignAdd<Float>) {
        optimizerAssignAddInitializers += initializer
    }

    /**
     * Returns a list of trainable variables used in layers.
     */
    public fun trainableLayerVariables(): List<Variable<Float>> {
        return layerVariables.filter { it.value }.keys.toList()
    }

    /**
     * Returns the variable's ability to be changed during the training.
     *
     * @param variableName Variable name.
     */
    public fun isVariableTrainable(variableName: String): Boolean {
        return layerVariables
            .filter { it.value } // only trainable
            .map { it.key.ref().op().name() } // extract names
            .any { it.contains(variableName) }
    }

    /**
     * Returns a list of non-trainable, 'frozen' variables used in layers.
     */
    public fun frozenLayerVariables(): List<Variable<Float>> {
        return layerVariables.filterNot { it.value }.keys.toList()
    }

    /**
     * Returns all variables used in all model layers.
     */
    public fun layerVariables(): List<Variable<Float>> {
        return layerVariables.keys.toList()
    }

    /**
     * Returns all variables used in optimizer and initialized by Assign TensorFlow operand.
     */
    public fun optimizerVariables(): List<Variable<Float>> {
        return optimizerVariables.toList()
    }

    /**
     * Initializes TensorFlow graph variables used in model layers.
     */
    public fun initializeGraphVariables(session: Session) {
        val runner = session.runner()
        // Only run the session if there is initializer ops.
        if (initializers.isNotEmpty()) {
            initializers.forEach {
                runner.addTarget(it.value as Operand<Float>)
            }
            runner.run()
        }
    }

    /**
     * Initializes TensorFlow graph variables used in optimizer.
     */
    public fun initializeOptimizerVariables(session: Session) {
        if (optimizerInitializers.isNotEmpty()) {
            optimizerInitializers.forEach {
                val runner = session.runner()
                runner.addTarget(it)
                runner.run()
            }

        }
        runAssignAddOpsForOptimizers(session)
    }

    private fun runAssignAddOpsForOptimizers(session: Session) {
        if (optimizerAssignAddInitializers.isNotEmpty()) {
            val runner = session.runner()

            optimizerAssignAddInitializers.forEach {
                runner.addTarget(it)
            }
            runner.run()
        }
    }
}
