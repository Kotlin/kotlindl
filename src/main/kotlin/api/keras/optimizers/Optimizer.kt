package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable

/** Base class for all optimizers. */
abstract class Optimizer<T : Number> {
    private lateinit var dtype: Class<T>

    /**
     * Top level map key is the variable name, lower level map key is the slot name.
     */
    private lateinit var slots: MutableMap<String, MutableMap<String, Variable<out T>>>

    fun prepareTargets(
        graph: KGraph<T>,
        tf: Ops,
        loss: Operand<T>
    ): List<Operand<T>> {
        val weights = graph.variables()

        slots = mutableMapOf()

        val gradients: Gradients = computeGradients(tf, loss, weights)

        val variableOutputs = variablesToOutputs(weights)

        createSlots(graph, tf, variableOutputs) // empty action if not overridden

        return applyGradients(graph, tf, weights, gradients)
    }

    private fun variablesToOutputs(variables: List<Variable<T>>): List<Output<out T>> {
        val variableOutputs: MutableList<Output<out T>> = mutableListOf()
        for (i in variables.indices) {
            variableOutputs.add(i, variables[i].asOutput())
        }

        return variableOutputs
    }

    protected abstract fun applyGradients(
        graph: KGraph<T>,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>>

    private fun computeGradients(
        tf: Ops,
        loss: Operand<T>,
        weights: List<Variable<T>>
    ): Gradients {
        return tf.gradients(loss, weights)
    }

    /**
     * No-op slot creation method.
     *
     * @param variables The variables to create slots for.
     */
    protected open fun createSlots(graph: KGraph<T>, tf: Ops, variables: List<Output<out T>>) {

    }

    abstract fun getOptimizerName(): String


    /**
     * Creates a slot in the graph for the specified variable with the specified name. Adds the slot's
     * initializer to the graph's initializers, and the slot to the Optimizer's slot map.
     *
     * @param variable    The variable to create the slot for.
     * @param slotName    The name of the slot.
     * @param initializer The initializer for the slot.
     * @param <T>         The type of the variable.
     */
    protected open fun createSlot(
        graph: KGraph<T>,
        tf: Ops,
        variable: Output<out T>,
        slotName: String,
        initializer: Operand<T>
    ) {
        val createName: String = createName(variable, slotName)
        val slot: Variable<T> = tf.withName(createName).variable(variable.shape(), getDType())
        val slotInit: Assign<T> = tf.assign(slot, initializer)

        graph.addOptimizerVariableInitializer(slotInit)

        val varName = variable.op().name()

        val variables: MutableMap<String, Variable<out T>> = slots.computeIfAbsent(slotName) { mutableMapOf() }
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
    ): Variable<T> {
        val variables: MutableMap<String, Variable<out T>> = slots[slotName]!!
        return variables[varName]!! as Variable<T>
    }

    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }

    open fun createName(variable: Output<out T>, slotName: String): String {
        return variable.op().name() + "-" + slotName
    }
}