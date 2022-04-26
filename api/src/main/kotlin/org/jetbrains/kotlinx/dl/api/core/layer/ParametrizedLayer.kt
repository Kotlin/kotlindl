/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.tensorflow.Session
import org.tensorflow.op.core.Variable

/**
 * Represents a [Layer] with parameters.
 */
public interface ParametrizedLayer {
    /** Variables used in this layer. */
    public val variables: List<KVariable>

    /** Number of parameters in this layer. */
    public val paramCount: Int
        get() = variables.sumOf { it.shape.numElements() }.toInt()
}

/**
 * Returns the number of parameters in this layer. If layer is not a [ParametrizedLayer], returns zero.
 */
public val Layer.paramCount: Int
    get() = if (this is ParametrizedLayer) paramCount else 0

/**
 * Returns all variables used in all layers.
 */
internal fun List<Layer>.variables(): List<KVariable> {
    return filterIsInstance<ParametrizedLayer>().flatMap { it.variables }
}

/**
 * Returns all variables used in the layers.
 */
internal fun List<Layer>.tfVariables(): List<Variable<Float>> = variables().map { it.variable }

/**
 * Returns a list of trainable variables used in the layers.
 */
internal fun List<Layer>.trainableVariables(): List<KVariable> {
    return filterIsInstance<TrainableLayer>().filter { it.isTrainable }.flatMap { it.variables }
}

/**
 * Returns a list of trainable variables used in the layers.
 */
internal fun List<Layer>.trainableTfVariables(): List<Variable<Float>> = trainableVariables().map { it.variable }

/**
 * Returns a list of non-trainable, 'frozen' variables used in the layers.
 */
internal fun List<Layer>.frozenTfVariables(): List<Variable<Float>> {
    return filterIsInstance<ParametrizedLayer>()
        .filter { it !is TrainableLayer || !it.isTrainable }
        .flatMap { it.variables }
        .map { it.variable }
}

/**
 * Initializes this layers variables using provided initializer operands.
 */
public fun ParametrizedLayer.initialize(session: Session) {
    // Only run the session if there is initializer ops.
    if (variables.isNotEmpty()) {
        val runner = session.runner()
        variables.map { it.initializerOperand }.forEach(runner::addTarget)
        runner.run()
    }
}

/**
 * Initializes variables for [ParametrizedLayer] instances using provided initializer operands.
 */
public fun List<Layer>.initializeVariables(session: Session) {
    filterIsInstance<ParametrizedLayer>().forEach { it.initialize(session) }
}