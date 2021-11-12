package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

public data class Parameter(
    val name: String,
    val shape: Shape,
    val tfVar: Variable<Float>,
    val initializer: Initializer?,
    val initOp: Operand<Float>,
    val regularizer: Regularizer?
)

public fun parameter(
    tf: Ops,
    kGraph: KGraph,
    variableName: String,
    isTrainable: Boolean,
    shape: Shape,
    fanIn: Int,
    fanOut: Int,
    initializer: Initializer,
    regularizer: Regularizer?
): Parameter {
    val tfVariable = tf.withName(variableName).variable(shape, getDType())

    val initOp = initializer.apply(fanIn, fanOut, tf, tfVariable, variableName)
    kGraph.addLayerVariable(tfVariable, isTrainable)
    kGraph.addInitializer(variableName, initOp)
    if (regularizer != null) kGraph.addVariableRegularizer(tfVariable, regularizer)

    return Parameter(
        name = variableName,
        shape = shape,
        tfVar = tfVariable,
        initializer = initializer,
        initOp = initOp,
        regularizer = regularizer
    )
}