/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Layer that adds a list of inputs.
 *
 * It takes as input a list of tensors, all the same shape, and returns a single tensor (also of the same shape).
 *
 * @property [layerTypeName] Specified layer name used for tf operation alias building.
 */
public abstract class AbstractMerge(public val layerTypeName: String, name: String = "") : Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {

    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        throw UnsupportedOperationException("This layer is not supported for Sequential model!")
    }

    override fun computeOutputShapeFromInboundLayers(): TensorShape {
        checkInputShapesOfInboundLayers() //TODO: crash efficientNet models
        outputShape = inboundLayers[0].outputShape.clone()
        return outputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return input
    }

    override fun forward(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        checkInputShapesOfInputOperands(input) //TODO: crash efficientNet models
        return tf.withName(layerTypeName).identity(mergeFunction(input, tf))
    }

    /** Should be overridden in all AbstractMerge descendants. */
    protected abstract fun mergeFunction(
        input: List<Operand<Float>>,
        tf: Ops
    ): Operand<Float>

    /** Checks input shapes of input operands. */
    protected open fun checkInputShapesOfInputOperands(input: List<Operand<Float>>) {
        require(input.size > 1) { "The number of input layers should be more than 1." }

        val firstInputShape = TensorShape(input[0].asOutput().shape())

        for (layer in input) {
            val tensorShape = TensorShape(
                layer.asOutput().shape()
            )
            require(
                firstInputShape == tensorShape
            ) { "The shape of first input $firstInputShape should be equal to the shape $tensorShape of $layer " }
        }
    }

    private fun checkInputShapesOfInboundLayers() {
        val firstInputShape = inboundLayers[0].outputShape

        for (layer in inboundLayers) {
            val tensorShape = layer.outputShape
            require(
                firstInputShape == tensorShape
            ) { "The shape of first input $firstInputShape should be equal to the shape $tensorShape of $layer " }
        }
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean
        get() = false
}
