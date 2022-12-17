/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.GraphTrainableModel
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.core.shape.tail
import org.jetbrains.kotlinx.dl.api.core.shape.toTensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Base abstract class for all layers.
 *
 * @property [name] Layer name. A new name is generated during model compilation when provided name is empty.
 */
public abstract class Layer(public var name: String) {
    /** Output data tensor shape. */
    public lateinit var outputShape: TensorShape

    /** Model where this layer is used. */
    public var parentModel: GraphTrainableModel? = null

    /** Returns inbound layers. */
    public var inboundLayers: MutableList<Layer> = mutableListOf()

    /** Returns outbound layers. */
    public var outboundLayers: MutableList<Layer> = mutableListOf()

    /**
     * Extend this function to define variables in the layer and compute layer output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [input] Layer input.
     * @param [isTraining] TensorFlow operand for switching between training and inference modes.
     * @param [numberOfLosses] TensorFlow operand for batch size data.
     */
    public abstract fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float>

    /**
     * Extend this function to define variables in the layer and compute layer output.
     *
     * NOTE: This function should be overridden for layers with multiple inputs.
     * NOTE: Used in Functional API
     *
     * @param [input] Layer input list.
     * @param [isTraining] TensorFlow operand for switching between training and inference modes.
     * @param [numberOfLosses] TensorFlow operand for batch size data.
     */
    public open fun build(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return build(tf, input.first(), isTraining, numberOfLosses)
    }

    /** Important part of functional API. It takes [layers] as input and saves them to the [inboundLayers] of the given layer. */
    public operator fun invoke(vararg layers: Layer): Layer {
        inboundLayers = layers.toMutableList()
        return this
    }

    /** Returns True, if layer has internal activation function. */
    public abstract val hasActivation: Boolean
}

internal fun requireArraySize(array: IntArray, size: Int, name: String) =
    require(array.size == size) {
        "$name is expected to have size equal $size but got ${array.size}"
    }

internal fun Layer.setOutputShape(shape: Shape) {
    check(shape.tail().all { elem -> elem > 0 })
    {
        "The last dimensions (except first = -1) of shape of layer $name contains zero or negative dimension values: ${shape}.\n" +
                "Analyze your model architecture and layer output shapes carefully to discover a problem."
    }
    outputShape = shape.toTensorShape()
}