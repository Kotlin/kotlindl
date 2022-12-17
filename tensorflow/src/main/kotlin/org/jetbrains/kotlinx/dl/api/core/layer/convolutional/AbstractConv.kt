/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.*
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.shape.shapeFromDims
import org.jetbrains.kotlinx.dl.api.core.util.toLongArray
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
 * @property [activation] activation used in [build] operation implementation
 * @property [kernelInitializer] initializer used in actual kernel variable filling implementation
 * @property [biasInitializer] initializer used in actual bias variable filling implementation
 * @property [kernelRegularizer] regularizer function used in actual kernel variable filling implementation
 * @property [biasRegularizer] regularizer function used in actual bias variable filling implementation
 * @property [activityRegularizer] regularizer function applied to the output of the layer
 * @property [padding] numbers to keep for the padding for implementation
 * @property [useBias] flag if bias should be used during actual [build] implementation
 * @constructor Creates [AbstractConv] object
 *
 * @param name of the layer to name its variables
 */
public abstract class AbstractConv(
    name: String
) : Layer(name), ParametrizedLayer {

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
    internal lateinit var kernel: KVariable

    /** Tensor with bias weights */
    internal var bias: KVariable? = null

    public override val variables: List<KVariable>
        get() = listOfNotNull(kernel, bias)

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val inputShape = input.asOutput().shape()
        // Amount of channels should be the last value in the inputShape
        val numberOfChannels = inputShape.size(inputShape.numDimensions() - 1)

        val inputDepth = numberOfChannels // number of input channels
        val outputDepth = getOutputDepth(numberOfChannels) // number of output channels
        val fanIn = (inputDepth * multiply(kernelSize.toLongArray())).toInt()
        val fanOut = ((outputDepth * multiply(kernelSize.toLongArray())).toDouble() /
                multiply(strides.toLongArray()).toDouble()).roundToInt()

        kernel = createVariable(
            tf,
            kernelVarName(name),
            computeKernelShape(numberOfChannels),
            fanIn,
            fanOut,
            kernelInitializer,
            kernelRegularizer
        )

        if (useBias) {
            bias = createVariable(
                tf,
                biasVarName(name),
                computeBiasShape(numberOfChannels),
                fanIn,
                fanOut,
                biasInitializer,
                biasRegularizer
            )
        }

        val convolution = convImplementation(tf, input)
        val withBias = bias?.let { tf.nn.biasAdd(convolution, it.variable) } ?: convolution
        return Activations.convert(activation).apply(tf, withBias, name)
    }

    override val hasActivation: Boolean get() = true

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
}

private fun multiply(values: LongArray) = values.fold(1L, Long::times)
