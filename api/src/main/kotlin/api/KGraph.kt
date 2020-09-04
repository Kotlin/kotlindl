package api

import org.tensorflow.Graph
import org.tensorflow.GraphOperation
import org.tensorflow.Operand
import org.tensorflow.Session
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.AssignAdd
import org.tensorflow.op.core.Variable

class KGraph(graphDef: ByteArray, prefix: String) : AutoCloseable {
    constructor(graphDef: ByteArray) : this(graphDef, "")

    var tfGraph: Graph = Graph()

    /** A list of initializer to initialize the trainableVariables. */
    private val optimizerInitializers: MutableList<Assign<*>> = mutableListOf()

    /** A list of initializer to initialize the trainableVariables. */
    private val optimizerAssignAddInitializers: MutableList<AssignAdd<*>> = mutableListOf()

    /** A list of variables to train. */
    private val layerVariables: MutableMap<Variable<Float>, Boolean> = mutableMapOf()

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

    override fun close() {
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

    fun addLayerVariable(variable: Variable<Float>, isTrainable: Boolean) {
        check(!layerVariables.contains(variable)) { "$variable is added to graph already. Analyze and fix the static graph building process." }
        layerVariables[variable] = isTrainable
    }

    fun addOptimizerVariable(variable: Variable<Float>) {
        check(!optimizerVariables.contains(variable)) { "$variable is added to graph already. Analyze and fix the static graph building process." }
        optimizerVariables.add(variable)
    }

    fun addInitializer(variableName: String, initializer: Assign<Float>) {
        check(!initializers.contains(variableName)) { "$variableName has initializer already. Analyze and fix the static graph building process." }
        initializers[variableName] = initializer
    }

    fun addOptimizerVariableInitializer(initializer: Assign<*>) {
        optimizerInitializers += initializer
    }

    fun addOptimizerVariableAssignAddInitializer(initializer: AssignAdd<Long>) {
        optimizerAssignAddInitializers += initializer
    }

    fun trainableLayerVariables(): List<Variable<Float>> {
        return layerVariables.filter { it.value }.keys.toList()
    }

    fun layerVariables(): List<Variable<Float>> {
        return layerVariables.keys.toList()
    }

    fun optimizerVariables(): List<Variable<Float>> {
        return optimizerVariables.toList()
    }

    fun initializeGraphVariables(session: Session) {
        val runner = session.runner()

        initializers.forEach {
            runner.addTarget(it.value as Operand<Float>)
        }

        runner.run()
    }

    fun initializeOptimizerVariables(session: Session) {
        if (optimizerInitializers.isNotEmpty()) {
            optimizerInitializers.forEach {
                val runner = session.runner()
                runner.addTarget(it as Operand<Float>)
                runner.run()
            }

        }
        runAssignAddOpsForOptimizers(session)
    }

    private fun runAssignAddOpsForOptimizers(session: Session) {
        if (optimizerAssignAddInitializers.isNotEmpty()) {
            val runner = session.runner()

            optimizerAssignAddInitializers.forEach {
                runner.addTarget(it as Operand<Float>)
            }
            runner.run()
        }
    }
}
