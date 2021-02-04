/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.initializer.Initializer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * Base abstract class for all layers.
 *
 * @param [name] Layer name. Would be changed if empty during model compilation.
 */
public abstract class Layer(public var name: String) {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could be changed during the training.
     */
    public var isTrainable: Boolean = true

    /** Output data tensor shape. */
    public lateinit var outputShape: LongArray

    /** Model where this layer is used. */
    public lateinit var parentModel: TrainableModel

    /** Returns number of input parameters. */
    protected var fanIn: Int = Int.MIN_VALUE

    /** Returns number of output parameters. */
    protected var fanOut: Int = Int.MIN_VALUE

    /** Returns inbound layers. */
    public var inboundLayers: List<Layer> = emptyList()

    /** Returns outbound layers. */
    public var outboundLayers: List<Layer> = emptyList()

    /**
     * Extend this function to define variables in layer.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     * @param [inputShape] Input shape, result of [computeOutputShape] call from previous layer.
     */
    public abstract fun build(tf: Ops, kGraph: KGraph, inputShape: Shape)

    /**
     * Computes output shape, based on [inputShape] and [Layer] type.
     */
    public abstract fun computeOutputShape(inputShape: Shape): Shape

    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public abstract fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float>

    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    public open fun forward(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return forward(tf, input[0], isTraining, numberOfLosses)
    }

    /**
     * Adds a new weight tensor to the layer
     *
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected fun addWeight(
        tf: Ops,
        kGraph: KGraph,
        name: String,
        variable: Variable<Float>,
        initializer: Initializer
    ): Variable<Float> {
        // require(fanIn != Int.MIN_VALUE) { "fanIn should be calculated before initialization for variable $name" }
        // require(fanOut != Int.MIN_VALUE) { "fanOut should be calculated before initialization for variable $name" }

        val initOp = initializer.apply(fanIn, fanOut, tf, variable, name)
        kGraph.addLayerVariable(variable, isTrainable)
        kGraph.addInitializer(name, initOp)
        return variable
    }

    public operator fun invoke(vararg layers: Layer): Layer {
        inboundLayers = layers.toList()
        return this
    }

    /** Returns layer's weights. */
    public abstract val weights: List<Array<*>>

    /** Returns True, if layer has internal activation function. */
    public abstract val hasActivation: Boolean

    /** Returns amount of neurons. */
    public abstract val paramCount: Int

    // apply


}
