/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import java.lang.IllegalArgumentException
import kotlin.math.roundToInt

public abstract class AbstractConv(
    protected val filtersInternal: Long,
    protected val kernelSizeInternal: LongArray,
    protected val stridesInternal: LongArray,
    protected val dilationsInternal: LongArray,
    protected val activationInternal: Activations,
    protected val kernelInitializerInternal: Initializer,
    protected val biasInitializerInternal: Initializer,
    protected val kernelRegularizerInternal: Regularizer?,
    protected val biasRegularizerInternal: Regularizer?,
    protected val activityRegularizerInternal: Regularizer?,
    protected val paddingInternal: ConvPadding,
    protected val useBiasInternal: Boolean,
    protected val kernelVariableName: String,
    protected val biasVariableName: String,
    name: String
) : Layer(name) {
    // weight tensors
    protected lateinit var kernel: Variable<Float>
    protected var bias: Variable<Float>? = null

    // weight tensor shapes
    protected lateinit var kernelShape: Shape
    protected lateinit var biasShape: Shape

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape
        val lastElement = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        kernelShape = shapeFromDims(*kernelSizeInternal, lastElement, filtersInternal)
        biasShape = Shape.make(filtersInternal)

        // should be calculated before addWeight because it's used in calculation,
        // need to rewrite addWeight to avoid strange behaviour calculate fanIn, fanOut
        val inputDepth = lastElement // number of input channels
        val outputDepth = filtersInternal // number of output channels

        fanIn = (inputDepth * multiply(*kernelSizeInternal)).toInt()
        fanOut = ((outputDepth * multiply(*kernelSizeInternal)).toDouble() /
                multiply(*stridesInternal).toDouble()).roundToInt()

        val (kernelVariableName, biasVariableName) = defineVariableNames()
        createConvVariables(tf, kernelVariableName, biasVariableName, kGraph)
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val shape = defineOutputShape(inputShape)
        outputShape = TensorShape(shape)
        return shape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        var output = convImplementation(tf, input)

        if (useBiasInternal) {
            output = tf.nn.biasAdd(output, bias)
        }

        val activated = Activations.convert(activationInternal).apply(tf, output, name)
        return activityRegularizerInternal?.apply(tf, activated) ?: activated
    }

    /** Returns the shape of kernel weights. */
    public val kernelShapeArray: LongArray get() = TensorShape(kernelShape).dims()

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray get() = TensorShape(biasShape).dims()

    override var weights: Map<String, Array<*>>
        get() = extractConvWeights()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = true

    override val paramCount: Int
        get() = (kernelShape.numElements() + biasShape.numElements()).toInt()

    private fun extractConvWeights(): Map<String, Array<*>> = extractWeights(defineVariableNames().toList())

    private fun defineVariableNames(): Pair<String, String> = if (name.isNotEmpty()) {
        Pair(kernelVarName(name), biasVarName(name))
    } else {
        Pair(kernelVariableName, biasVariableName)
    }

    private fun createConvVariables(
        tf: Ops,
        kernelVariableName: String,
        biasVariableName: String,
        kGraph: KGraph
    ) {
        kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
        kernel = addWeight(tf, kGraph, kernelVariableName, kernel, kernelInitializerInternal, kernelRegularizerInternal)

        if (useBiasInternal) {
            bias = tf.withName(biasVariableName).variable(biasShape, getDType())
            bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializerInternal, biasRegularizerInternal)
        }
    }

    protected abstract fun kernelVarName(name: String): String

    protected abstract fun biasVarName(name: String): String

    protected abstract fun convImplementation(tf: Ops, input: Operand<Float>): Operand<Float>

    protected abstract fun defineOutputShape(inputShape: Shape): Shape
}

internal fun assertArraySize(array: LongArray, size: Int, name: String) {
    if (array.size != size) {
        throw IllegalArgumentException("$name is expected to have size equal $size")
    }
}

private fun multiply(vararg values: Long) = values.fold(1L, Long::times)
