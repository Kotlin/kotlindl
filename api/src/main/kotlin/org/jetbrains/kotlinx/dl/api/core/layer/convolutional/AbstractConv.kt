/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.KVariable
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.createVariable
import org.jetbrains.kotlinx.dl.api.core.layer.toLongArray
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import kotlin.math.roundToInt

/**
 * Abstract Convolutional layer is a base block for building base types of convolutional layers
 * of any dimensionality. It should simplify the internal calculations needed in most convolutional
 * layers and abstract the naming weights for these layers. It keeps the actual implementation
 * of convolutional layers, i.e., the kernel and bias learnable variables that should
 * be used in child classes in actual implementations of these layers.
 *
 * @property [filters] number used by default in calculation of layer weights and i/o shapes
 * @property [kernelSize] numbers used by default in calculation of layer weights and i/o shapes
 * @property [strides] numbers used by default in calculation of layer weights and i/o shapes
 * @property [dilations] numbers to keep for the dilations for implementation
 * @property [activation] activation used in [forward] operation implementation
 * @property [kernelInitializer] initializer used in actual kernel variable filling implementation
 * @property [biasInitializer] initializer used in actual bias variable filling implementation
 * @property [kernelRegularizer] regularizer function used in actual kernel variable filling implementation
 * @property [biasRegularizer] regularizer function used in actual bias variable filling implementation
 * @property [activityRegularizer] regularizer function applied to the output of the layer
 * @property [padding] numbers to keep for the padding for implementation
 * @property [useBias] flag if bias should be used during actual [forward] implementation
 * @constructor Creates [AbstractConv] object
 *
 * @param name of the layer to name its variables
 */
public abstract class AbstractConv(
    name: String
) : Layer(name) {
    
    protected abstract val filters: Int
    protected abstract val kernelSize: IntArray
    protected abstract val strides: IntArray
    protected abstract val dilations: IntArray
    protected abstract val activation: Activations
    protected abstract val kernelInitializer: Initializer
    protected abstract val biasInitializer: Initializer
    protected abstract val kernelRegularizer: Regularizer?
    protected abstract val biasRegularizer: Regularizer?
    protected abstract val activityRegularizer: Regularizer?
    protected abstract val padding: ConvPadding
    internal abstract val useBias: Boolean

    /** Tensor with kernel weights */
    protected lateinit var kernel: KVariable

    /** Tensor with bias weights */
    protected var bias: KVariable? = null

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        val inputDepth = numberOfChannels // number of input channels
        val outputDepth = getOutputDepth(numberOfChannels) // number of output channels
        val fanIn = (inputDepth * multiply(kernelSize.toLongArray())).toInt()
        val fanOut = ((outputDepth * multiply(kernelSize.toLongArray())).toDouble() /
                     multiply(strides.toLongArray()).toDouble()).roundToInt()

        kernel = createVariable(
            tf,
            kGraph,
            kernelVarName(name),
            isTrainable,
            computeKernelShape(numberOfChannels),
            fanIn,
            fanOut,
            kernelInitializer,
            kernelRegularizer
        )

        if (useBias) {
            bias = createVariable(
                tf,
                kGraph,
                biasVarName(name),
                isTrainable,
                computeBiasShape(numberOfChannels),
                fanIn,
                fanOut,
                biasInitializer,
                biasRegularizer
            )
        }
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
        val convolution = convImplementation(tf, input)

        val withBias = bias?.let { tf.nn.biasAdd(convolution, it.variable) } ?: convolution

        return Activations.convert(activation).apply(tf, withBias, name)
    }

    /** Returns the shape of kernel weights. */
    public val kernelShapeArray: LongArray? get() = if (this::kernel.isInitialized)  TensorShape(kernel.shape).dims() else null

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray? get() = bias?.let { TensorShape(it.shape).dims() }

    override var weights: Map<String, Array<*>>
        get() = extractWeights(kernel, bias)
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = true

    override val paramCount: Int
        get() = (kernel.shape.numElements() + (bias?.shape?.numElements() ?: 0)).toInt()

    /** Define the number of output channels given the number of input channels.
     *  Defaults to the number of filter in convolutional layer. */
    protected open fun getOutputDepth(numberOfChannels: Long): Long = filters.toLong()

    /**
     * Define the [kernel] shape by default from its [kernelSize],
     * [filters] and the given [numberOfChannels] from input Tensor.
     *
     * @param numberOfChannels for input of this layer
     */
    protected open fun computeKernelShape(numberOfChannels: Long): Shape =
        shapeFromDims(*kernelSize.toLongArray(), numberOfChannels, filters.toLong())

    /**
     * Define the [bias] shape by default from its [filters] and
     * the given [numberOfChannels] from input Tensor.
     *
     * @param numberOfChannels for input of this layer
     */
    protected open fun computeBiasShape(numberOfChannels: Long): Shape =
        Shape.make(filters.toLong())

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
}

private fun multiply(values: LongArray) = values.fold(1L, Long::times)
