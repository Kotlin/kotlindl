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

/**
 * Abstract Convolutional layer is a base block for building base types of convolutional layers
 * of any dimensionality. It should simplify the internal calculations needed in most of
 * the convolutional layers and abstract the process of naming weights for these layers. It keeps
 * the actual implementation of convolutional layers e.i. the kernel and bias learnable variables
 * that should be used in child classes in actual implementations of these layers. If the child class
 * uses some values for its implementation in other form than it is kept in this child class,
 * then this abstract class `internal` properties should keep the implementation values
 * while the child class properties should keep the printable values that are more representative.
 * But in most cases the `internal` and child values will be the same.
 *
 * @property filtersInternal number used by default in calculation of layer weights and i/o shapes
 * @property kernelSizeInternal numbers used by default in calculation of layer weights and i/o shapes
 * @property stridesInternal numbers used by default in calculation of layer weights and i/o shapes
 * @property dilationsInternal numbers to keep for the dilations for implementation
 * @property activationInternal activation used in [forward] operation implementation
 * @property kernelInitializerInternal kernelInitializer used in actual kernel variable filling implementation
 * @property biasInitializerInternal biasInitializer used in actual bias variable filling implementation
 * @property kernelRegularizerInternal kernelRegularizer used in actual kernel variable filling implementation
 * @property biasRegularizerInternal biasRegularizer used in actual bias variable filling implementation
 * @property activityRegularizerInternal regularizer function applied to the output of the layer
 * @property paddingInternal numbers to keep for the padding for implementation
 * @property useBiasInternal flag if bias should be used during actual [forward] implementation
 * @property kernelVariableName name of kernel used when no layer name is defined
 * @property biasVariableName name of bias used when no layer name is defined
 * @constructor Creates [AbstractConv] object
 *
 * @param name of the layer to name its variables
 */
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

    /** Tensor with learnable variables for kernel defined by internal shapes */
    protected lateinit var kernel: Variable<Float>

    /** Tensor with learnable variables for bias defined by internal shapes */
    protected var bias: Variable<Float>? = null

    /** Shape of internal implementation of kernel variable */
    protected lateinit var biasShape: Shape

    /** Shape of internal implementation of bias variable */
    protected lateinit var kernelShape: Shape

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        computeMatricesShapes(numberOfChannels)

        // should be calculated before addWeight because it's used in calculation
        val inputDepth = numberOfChannels // number of input channels
        val outputDepth = getOutputDepth(numberOfChannels) // number of output channels
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

        return Activations.convert(activationInternal).apply(tf, output, name)
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

    /** Define the number of output channels given the number of input channels.
     *  Defaults to the number of filter in convolutional layer. */
    protected open fun getOutputDepth(numberOfChannels: Long): Long = filtersInternal

    /**
     * Define the [kernelShape] and [biasShape] by default from its [kernelSizeInternal],
     * [filtersInternal], [filtersInternal] and the given [numberOfChannels] from input Tensor.
     *
     * @param numberOfChannels for input of this layer
     */
    protected open fun computeMatricesShapes(numberOfChannels: Long) {
        kernelShape = shapeFromDims(*kernelSizeInternal, numberOfChannels, filtersInternal)
        biasShape = Shape.make(filtersInternal)
    }

    /** Given a layer name specify its kernel name. */
    protected abstract fun kernelVarName(name: String): String

    /** Given a layer name specify its bias name. */
    protected abstract fun biasVarName(name: String): String

    /** The actual layer operation implementation without adding the bias which is added by the abstract class. */
    protected abstract fun convImplementation(tf: Ops, input: Operand<Float>): Operand<Float>

    /**
     * Actual implementation of [computeOutputShape] which only defines the value
     * of output shape without the need of saving it to some variable.
     *
     * @param inputShape which can be used to define the output shape
     * @return the defined output shape that is saved in class variable and returned by [computeOutputShape]]
     */
    protected abstract fun defineOutputShape(inputShape: Shape): Shape

    /** Extract weights of the layer with the names from [defineVariableNames]. */
    private fun extractConvWeights(): Map<String, Array<*>> = extractWeights(defineVariableNames().toList())

    /** Create the names of variables of the layer based on layer name or not if not present. */
    private fun defineVariableNames(): Pair<String, String> = if (name.isNotEmpty()) {
        Pair(kernelVarName(name), biasVarName(name))
    } else {
        Pair(kernelVariableName, biasVariableName)
    }

    /** Create the variables of the layer in proper order. */
    private fun createConvVariables(
        tf: Ops,
        kernelVariableName: String,
        biasVariableName: String,
        kGraph: KGraph
    ) {
        kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
        if (useBiasInternal) bias = tf.withName(biasVariableName).variable(biasShape, getDType())

        kernel = addWeight(tf, kGraph, kernelVariableName, kernel, kernelInitializerInternal, kernelRegularizerInternal)
        if (useBiasInternal) bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializerInternal, biasRegularizerInternal)
    }
}

private fun multiply(vararg values: Long) = values.fold(1L, Long::times)
