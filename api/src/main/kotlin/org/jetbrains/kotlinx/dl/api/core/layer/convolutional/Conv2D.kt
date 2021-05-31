/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.*
import org.jetbrains.kotlinx.dl.api.core.util.conv2dBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.conv2dKernelVarName
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import org.tensorflow.op.nn.Conv2d
import org.tensorflow.op.nn.Conv2d.dilations
import java.lang.IllegalArgumentException
import kotlin.math.roundToInt

private const val KERNEL_VARIABLE_NAME = "conv2d_kernel"

private const val BIAS_VARIABLE_NAME = "conv2d_bias"

/**
 * 2D convolution layer (e.g. spatial convolution over images).
 *
 * This layer creates a convolution kernel that is convolved (actually cross-correlated)
 * with the layer input to produce a tensor of outputs.
 * Finally, the `activation` is applied to the outputs as well.
 *
 * It expects input data of size `(N, H, W, C)` where
 * ```
 * N - batch size
 * H - height
 * W - width
 * C - number of channels
 * ```
 *
 * @property [filters] The dimensionality of the output space (i.e. the number of filters in the convolution).
 * @property [kernelSize] Two long numbers, specifying the height and width of the 2D convolution window.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * NOTE: Specifying any stride value != 1 is incompatible with specifying any `dilations` value != 1.
 * @property [dilations] Four numbers, specifying the dilation rate to use for dilated convolution for each dimension of input tensor.
 * @property [activation] Activation function.
 * @property [kernelInitializer] An initializer for the convolution kernel
 * @property [biasInitializer] An initializer for the bias vector.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @property [useBias] If true the layer uses a bias vector.
 * @constructor Creates [Conv2D] object.
 */
public class Conv2D(
    public val filters: Long = 32,
    public val kernelSize: LongArray = longArrayOf(3, 3),
    public val strides: LongArray = longArrayOf(1, 1, 1, 1),
    public val dilations: LongArray = longArrayOf(1, 1, 1, 1),
    public val activation: Activations = Activations.Relu,
    public val kernelInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val padding: ConvPadding = ConvPadding.SAME,
    public val useBias: Boolean = true,
    name: String = ""
) : Conv2DImpl(
    filtersInternal = filters,
    kernelSizeInternal = kernelSize,
    stridesInternal = strides,
    dilationsInternal = dilations,
    activationInternal = activation,
    kernelInitializerInternal = kernelInitializer,
    biasInitializerInternal = biasInitializer,
    paddingInternal = padding,
    useBiasInternal = useBias,
    kernelVariableName = KERNEL_VARIABLE_NAME,
    biasVariableName = BIAS_VARIABLE_NAME,
    name = name
) {
    init {
        assertArraySize(kernelSize, 2, "kernelSize")
        assertArraySize(strides, 4, "strides")
        assertArraySize(dilations, 4, "dilations")
    }

    override fun toString(): String {
        return "Conv2D(filters=$filters, kernelSize=${kernelSize.contentToString()}, strides=${strides.contentToString()}, " +
                "dilations=${dilations.contentToString()}, activation=$activation, kernelInitializer=$kernelInitializer, " +
                "biasInitializer=$biasInitializer, kernelShape=$kernelShape, biasShape=$biasShape, padding=$padding)"
    }
}

public abstract class Conv2DImpl(
    private val filtersInternal: Long,
    private val kernelSizeInternal: LongArray,
    private val stridesInternal: LongArray,
    private val dilationsInternal: LongArray,
    private val activationInternal: Activations,
    private val kernelInitializerInternal: Initializer,
    private val biasInitializerInternal: Initializer,
    private val paddingInternal: ConvPadding,
    private val useBiasInternal: Boolean,
    private val kernelVariableName: String,
    private val biasVariableName: String,
    name: String = ""
) : Layer(name) {
    // weight tensors
    private lateinit var kernel: Variable<Float>
    private var bias: Variable<Float>? = null

    // weight tensor shapes
    protected lateinit var kernelShape: Shape
    protected lateinit var biasShape: Shape

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        // Amount of channels should be the last value in the inputShape (make warning here)
        val lastElement = inputShape.size(inputShape.numDimensions() - 1)

        // Compute shapes of kernel and bias matrices
        kernelShape = shapeFromDims(*kernelSizeInternal, lastElement, filtersInternal)
        biasShape = Shape.make(filtersInternal)

        // should be calculated before addWeight because it's used in calculation,
        // need to rewrite addWeight to avoid strange behaviour calculate fanIn, fanOut
        val inputDepth = lastElement // amount of channels
        val outputDepth = filtersInternal // amount of channels for the next layer

        fanIn = (inputDepth * kernelSizeInternal[0] * kernelSizeInternal[1]).toInt()
        fanOut = ((outputDepth * kernelSizeInternal[0] * kernelSizeInternal[1] /
                (stridesInternal[0].toDouble() * stridesInternal[1])).roundToInt())

        val (kernelVariableName, biasVariableName) = defineVariableNames()
        createConv2DVariables(tf, kernelVariableName, biasVariableName, kGraph)
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = convOutputLength(
            rows, kernelSizeInternal[0].toInt(), paddingInternal,
            stridesInternal[1].toInt(), dilationsInternal[1].toInt()
        )
        cols = convOutputLength(
            cols, kernelSizeInternal[1].toInt(), paddingInternal,
            stridesInternal[2].toInt(), dilationsInternal[2].toInt()
        )

        val shape = Shape.make(inputShape.size(0), rows, cols, filtersInternal)
        outputShape = TensorShape(shape)
        return shape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingName = paddingInternal.paddingName
        val options: Conv2d.Options = dilations(dilationsInternal.toList()).dataFormat("NHWC")
        var output: Operand<Float> = tf.nn.conv2d(input, kernel, stridesInternal.toMutableList(), paddingName, options)

        if (useBiasInternal) {
            output = tf.nn.biasAdd(output, bias)
        }

        return Activations.convert(activationInternal).apply(tf, output, name)
    }

    /** Returns the shape of kernel weights. */
    public val kernelShapeArray: LongArray get() = TensorShape(kernelShape).dims()

    /** Returns the shape of bias weights. */
    public val biasShapeArray: LongArray get() = TensorShape(biasShape).dims()

    override val weights: Map<String, Array<*>> get() = extractConv2DWeights()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int
        get() = (kernelShape.numElements() + biasShape.numElements()).toInt()

    private fun extractConv2DWeights(): Map<String, Array<*>> {
        return extractWeights(defineVariableNames().toList())
    }

    private fun defineVariableNames(): Pair<String, String> {
        return if (name.isNotEmpty()) {
            Pair(conv2dKernelVarName(name), conv2dBiasVarName(name))
        } else {
            Pair(kernelVariableName, biasVariableName)
        }
    }

    private fun createConv2DVariables(
        tf: Ops,
        kernelVariableName: String,
        biasVariableName: String,
        kGraph: KGraph
    ) {
        kernel = tf.withName(kernelVariableName).variable(kernelShape, getDType())
        if (useBiasInternal) bias = tf.withName(biasVariableName).variable(biasShape, getDType())

        kernel = addWeight(tf, kGraph, kernelVariableName, kernel, kernelInitializerInternal)
        if (useBiasInternal) bias = addWeight(tf, kGraph, biasVariableName, bias!!, biasInitializerInternal)
    }
}

private fun assertArraySize(array: LongArray, size: Int, name: String) {
    if (array.size != size) {
        throw IllegalArgumentException("$name is expected to have size equal $size")
    }
}