/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.TrainableModel
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.extension.convertTensorToMultiDimArray
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

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
    public lateinit var outputShape: TensorShape

    /** Model where this layer is used. */
    public var parentModel: TrainableModel? = null

    /** Returns inbound layers. */
    public var inboundLayers: MutableList<Layer> = mutableListOf()

    /** Returns outbound layers. */
    public var outboundLayers: MutableList<Layer> = mutableListOf()

    /**
     * Extend this function to define variables in layer.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     * @param [inputShape] Input shape, result of [computeOutputShape] call from previous layer.
     */
    public abstract fun build(tf: Ops, kGraph: KGraph, inputShape: Shape)


    /**
     * Extend this function to define variables in layer.
     *
     * NOTE: This function should be overridden for layers with multiple inputs.
     * NOTE: Used in Functional API
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     */
    public fun buildFromInboundLayers(tf: Ops, kGraph: KGraph) {
        require(inboundLayers.isNotEmpty()) { "There is no inbound layers to compute output shape" }
        build(tf, kGraph, inboundLayers[0].outputShape.toShape())
    }

    /**
     * Computes output shape, based on [inputShape] and [Layer] type.
     */
    public abstract fun computeOutputShape(inputShape: Shape): Shape

    /**
     * Computes output shape, based on input shapes of inbound layers.
     *
     * NOTE: This function should be overridden for layers with multiple inputs.
     * NOTE: Used in Functional API
     */
    public open fun computeOutputShapeFromInboundLayers(): TensorShape {
        require(inboundLayers.isNotEmpty()) { "There is no inbound layers to compute output shape" }
        return TensorShape(computeOutputShape(inboundLayers[0].outputShape.toShape()))
    }

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

    /** Important part of functional API. It takes [layers] as input and saves them to the [inboundLayers] of the given layer. */
    public operator fun invoke(vararg layers: Layer): Layer {
        inboundLayers = layers.toMutableList()
        return this
    }

    /** Extract weights values for provided variables. */
    protected fun extractWeights(vararg variables: KVariable?): Map<String, Array<*>> {
        require(parentModel != null) { "Layer $name is not related to any model!" }

        val session = parentModel!!.session
        val runner = session.runner()

        val variableNames = variables.mapNotNull { it?.name }
        for (variableName in variableNames) {
            runner.fetch(variableName)
        }

        val weights = runner.run().map { it.convertTensorToMultiDimArray() }
        return variableNames.zip(weights).toMap()
    }

    /** Extract weights values by variable names. */
    protected fun assignWeights(weights: Map<String, Array<*>>) {
        require(parentModel != null) { "Layer $name is not related to any model!" }

        for (variableName in weights.keys) {
            parentModel!!.assignVariable(variableName, weights[variableName]!!)
        }
    }

    /** Layer's weights. */
    public abstract var weights: Map<String, Array<*>>

    /** Returns True, if layer has internal activation function. */
    public abstract val hasActivation: Boolean

    /** Returns amount of neurons. */
    public abstract val paramCount: Int
}

internal fun requireArraySize(array: IntArray, size: Int, name: String) =
    require(array.size == size) {
        "$name is expected to have size equal $size but got ${array.size}"
    }

internal fun IntArray.toLongList(): List<Long> {
    return when (size) {
        0 -> emptyList()
        1 -> listOf(this[0].toLong())
        else -> this.mapTo(ArrayList(size)) { it.toLong() }
    }
}

internal fun IntArray.toLongArray(): LongArray {
    return when (size) {
        0 -> longArrayOf()
        1 -> longArrayOf(this[0].toLong())
        else -> LongArray(size) { this[it].toLong() }
    }
}