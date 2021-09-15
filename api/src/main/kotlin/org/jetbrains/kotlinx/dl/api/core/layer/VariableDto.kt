package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

public data class VariableDto(
    val name: String,
    val shape: Shape,
    val variable: Variable<Float>,
    val initializer: Initializer?,
    val initOp: Operand<Float>,
    val regularizer: Regularizer?
)

public fun variable(
    tf: Ops,
    variableName: String,
    shape: Shape,
    fanIn: Int,
    fanOut: Int,
    initializer: Initializer,
    regularizer: Regularizer?
): VariableDto {
    val tfVariable = tf.withName(variableName).variable(shape, getDType())
    return VariableDto(
        name = variableName,
        shape = shape,
        variable = tfVariable,
        initializer = initializer,
        initOp = initializer.apply(fanIn, fanOut, tf, tfVariable, variableName),
        regularizer = regularizer
    )
}