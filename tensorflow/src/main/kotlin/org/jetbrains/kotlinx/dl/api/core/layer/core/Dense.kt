/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.core

import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeUniform
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.jetbrains.kotlinx.dl.api.core.layer.KVariable
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.TrainableLayer
import org.jetbrains.kotlinx.dl.api.core.layer.createVariable
import org.jetbrains.kotlinx.dl.api.core.regularizer.Regularizer
import org.jetbrains.kotlinx.dl.api.core.util.denseBiasVarName
import org.jetbrains.kotlinx.dl.api.core.util.denseKernelVarName
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Densely-connected (fully-connected) layer class.
 *
 * This layer implements the operation:
 * `outputs = activation(inputs * kernel + bias)`
 *
 * where `activation` is the element-wise activation function
 * passed as the `activation` argument, `kernel` is a weights' matrix
 * created by the layer, and `bias` is a bias vector created by the layer
 * (only applicable if `use_bias` is `True`).
 *
 * @property [outputSize] Dimensionality of the output space.
 * @property [activation] Activation function.
 * @property [kernelInitializer] Initializer function for the 'kernel' weights matrix.
 * @property [biasInitializer] Initializer function for the bias.
 * @property [kernelRegularizer] Regularizer function applied to the `kernel` weights matrix.
 * @property [biasRegularizer] Regularizer function applied to the `bias` vector.
 * @property [activityRegularizer] Regularizer function applied to the output of the layer (its "activation").
 * @property [useBias] If true the layer uses a bias vector.
 * @property [name] Custom layer name.
 * @constructor Creates [Dense] object.
 */
public class Dense(
    public val outputSize: Int = 128,
    public val activation: Activations = Activations.Relu,
    public val kernelInitializer: Initializer = HeNormal(),
    public val biasInitializer: Initializer = HeUniform(),
    public val kernelRegularizer: Regularizer? = null,
    public val biasRegularizer: Regularizer? = null,
    public val activityRegularizer: Regularizer? = null,
    public val useBias: Boolean = true,
    name: String = ""
) : Layer(name), TrainableLayer {
    internal lateinit var kernel: KVariable
    internal var bias: KVariable? = null

    override val variables: List<KVariable>
        get() = listOfNotNull(kernel, bias)

    override var isTrainable: Boolean = true

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val fanIn = inputShape.size(inputShape.numDimensions() - 1).toInt()
        val fanOut = outputSize

        val kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), outputSize.toLong())
        kernel = createVariable(
            tf,
            denseKernelVarName(name),
            kernelShape,
            fanIn,
            fanOut,
            kernelInitializer,
            kernelRegularizer
        )

        if (useBias) {
            val biasShape = Shape.make(outputSize.toLong())
            bias = createVariable(
                tf,
                denseBiasVarName(name),
                biasShape,
                fanIn,
                fanOut,
                biasInitializer,
                biasRegularizer
            )
        }

        val matMul = tf.linalg.matMul(input, kernel.variable)
        val signal = bias?.let { tf.math.add(matMul, it.variable) } ?: matMul
        return Activations.convert(activation).apply(tf, signal, name)
    }

    override fun toString(): String {
        return "Dense(name = $name, isTrainable=$isTrainable, outputSize=$outputSize, activation=$activation, " +
                "kernelInitializer=$kernelInitializer, biasInitializer=$biasInitializer, " +
                "kernelRegularizer=$kernelRegularizer, biasRegularizer=$biasRegularizer, activityRegularizer=$activityRegularizer, " +
                "useBias=$useBias, hasActivation=$hasActivation, kernelShapeArray=${kernel.shape}, biasShapeArray=${bias?.shape})"
    }

    override val hasActivation: Boolean get() = true
}
