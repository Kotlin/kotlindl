/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * A class that keeps information about a single parameter of the [Layer].
 *
 * @property [name] name of the variable
 * @property [shape] shape of the variable
 * @property [variable] corresponding [Variable] object
 * @property [initializerOperand] variable initializer
 * @property [regularizer] variable regularizer
 */
public data class KVariable(
    val name: String,
    val shape: Shape,
    val variable: Variable<Float>,
    val initializerOperand: Operand<Float>,
    val regularizer: Regularizer?
)

internal fun createVariable(
    tf: Ops,
    kGraph: KGraph,
    variableName: String,
    isTrainable: Boolean,
    shape: Shape,
    fanIn: Int,
    fanOut: Int,
    initializer: Initializer,
    regularizer: Regularizer?
): KVariable {
    val tfVariable = tf.withName(variableName).variable(shape, getDType())

    val initOp = initializer.apply(fanIn, fanOut, tf, tfVariable, variableName)
    kGraph.addLayerVariable(tfVariable, isTrainable)
    kGraph.addInitializer(variableName, initOp)
    if (regularizer != null) kGraph.addVariableRegularizer(tfVariable, regularizer)

    return KVariable(
        name = variableName,
        shape = shape,
        variable = tfVariable,
        initializerOperand = initOp,
        regularizer = regularizer
    )
}