/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.optimizer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.util.defaultAssignOpName
import org.jetbrains.kotlinx.dl.api.core.util.defaultOptimizerVariableName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable

/**
 * Base class for all optimizers.
 *
 * @property [clipGradient] Strategy of gradient clipping as subclass of [ClipGradientAction].
 */
public abstract class Optimizer(public val clipGradient: ClipGradientAction) {
    /**
     * Top level map key is the variable name, lower level map key is the slot name.
     */
    private lateinit var slots: MutableMap<String, MutableMap<String, Variable<Float>>>

    /**
     * Prepares targets for optimization process.
     *
     * NOTE: Developer API.
     *
     * @param [graph] KGraph to be updated.
     * @param [tf] TensorFlow graph API for building operations.
     * @param [loss] Loss function.
     * @return List of optimizer operands to update variables.
     */
    internal fun prepareTargets(
        graph: KGraph,
        weights: List<Variable<Float>>,
        tf: Ops,
        loss: Operand<Float>
    ): List<Operand<Float>> {
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

    /**
     * Applies gradients to weights.
     *
     * NOTE: Developer API. Override this method in each optimizer.
     *
     * @param [graph] KGraph to be updated.
     * @param [tf] TensorFlow graph API for building operations.
     * @param [weights] Variables to update in optimizer.
     * @param [gradients] See [Gradients] for more information.
     */
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

    /** Returns optimizer name. */
    public abstract val optimizerName: String

    /**
     * Creates a slot in the graph for the specified variable with the specified name. Adds the slot's
     * initializer to the graph's initializers, and the slot to the optimiser's slot map.
     *
     * @param [graph] KGraph to be updated.
     * @param [tf] TensorFlow graph API for building operations.
     * @param [variable]    The variable to create the slot for.
     * @param [slotName]    The name of the slot.
     * @param [initializer] The initializer for the slot.
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
     * @param [varName]  The variable to lookup.
     * @param [slotName] The slot name.
     * @return The slot.
     */
    protected fun getSlot(
        varName: String,
        slotName: String
    ): Variable<Float> {
        val variables: MutableMap<String, Variable<Float>> = slots[slotName]!!
        return variables[varName]!!
    }

    /**
     * Creates name for [variable] used in slot with name [slotName].
     */
    internal open fun createName(variable: Output<Float>, slotName: String): String {
        return defaultOptimizerVariableName(variable.op().name() + "-" + slotName)
    }

    /** True, if optimizer is implemented for GPU. */
    internal abstract val isRunningOnGPU: Boolean
}
