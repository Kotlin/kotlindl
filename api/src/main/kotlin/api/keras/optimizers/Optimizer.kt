package api.keras.optimizers

import api.KGraph
import api.defaultAssignOpName
import api.defaultOptimizerVariableName
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable

/** Base class for all optimizers. */
abstract class Optimizer(val clipGradient: ClipGradientAction) {
    private lateinit var dtype: Class<Float>

    /**
     * Top level map key is the variable name, lower level map key is the slot name.
     */
    private lateinit var slots: MutableMap<String, MutableMap<String, Variable<Float>>>

    fun prepareTargets(
        graph: KGraph,
        tf: Ops,
        loss: Operand<Float>
    ): List<Operand<Float>> {
        val weights = graph.trainableLayerVariables()

        slots = mutableMapOf()

        val gradients: Gradients = computeGradients(tf, loss, weights)

        val variableOutputs = variablesToOutputs(weights)

        createSlots(graph, tf, variableOutputs) // empty action if not overridden

        return applyGradients(graph, tf, weights, gradients)
    }

    private fun variablesToOutputs(variables: List<Variable<Float>>): List<Output<Float>> {
        val variableOutputs: MutableList<Output<Float>> = mutableListOf()
        for (i in variables.indices) {
            variableOutputs.add(i, variables[i].asOutput())
        }

        return variableOutputs
    }

    protected abstract fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>>

    private fun computeGradients(
        tf: Ops,
        loss: Operand<Float>,
        weights: List<Variable<Float>>
    ): Gradients {
        return tf.gradients(loss, weights)
    }

    /**
     * No-op slot creation method.
     *
     * @param variables The variables to create slots for.
     */
    protected open fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {

    }

    abstract fun getOptimizerName(): String


    /**
     * Creates a slot in the graph for the specified variable with the specified name. Adds the slot's
     * initializer to the graph's initializers, and the slot to the Optimizer's slot map.
     *
     * @param variable    The variable to create the slot for.
     * @param slotName    The name of the slot.
     * @param initializer The initializer for the slot.
     * @param          The type of the variable.
     */
    protected open fun createSlot(
        graph: KGraph,
        tf: Ops,
        variable: Output<Float>,
        slotName: String,
        initializer: Operand<Float>
    ) {
        val createName: String = createName(variable, slotName)
        val slot: Variable<Float> = tf.withName(createName).variable(variable.shape(), getDType())

        val assignName = defaultAssignOpName(createName(variable, slotName))
        val slotInit: Assign<Float> = tf.withName(assignName).assign(slot, initializer)

        graph.addOptimizerVariableInitializer(slotInit)
        graph.addOptimizerVariable(slot)

        val varName = variable.op().name()

        val variables: MutableMap<String, Variable<Float>> = slots.computeIfAbsent(slotName) { mutableMapOf() }
        variables[varName] = slot
    }

    /**
     * Gets the slot associated with the specified variable and slot name.
     *
     * @param varName  The variable to lookup.
     * @param slotName The slot name.
     * @return The slot.
     */
    protected fun getSlot(
        varName: String,
        slotName: String
    ): Variable<Float> {
        val variables: MutableMap<String, Variable<Float>> = slots[slotName]!!
        return variables[varName]!!
    }

    fun getDType(): Class<Float> {
        return Float::class.javaObjectType
    }

    open fun createName(variable: Output<Float>, slotName: String): String {
        return defaultOptimizerVariableName(variable.op().name() + "-" + slotName)
    }
}