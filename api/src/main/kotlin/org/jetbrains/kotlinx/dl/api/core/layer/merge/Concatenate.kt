/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Layer that concatenates a list of inputs.
 *
 * It takes as input a list of tensors, all the same shape except
 * for the concatenation axis, and returns a single tensor that is the concatenation of all inputs.
 *
 * @property [axis] Axis along which to concatenate.
 */
public class Concatenate(
    public var axis: Int = 3,
    name: String = ""
) : AbstractMerge("ConcatenateLayer", name), NoGradients {
    override fun computeOutputShapeFromInboundLayers(): TensorShape {
        val inputShapes = mutableListOf<TensorShape>()
        inboundLayers.forEach { inboundLayer -> inputShapes.add(inboundLayer.outputShape) }
        val newShape = inputShapes[0].clone()

        var axe = axis
        if (axis == -1) { // it influences on nasmobilemodel
            val rank: Int = inputShapes[0].rank()
            axe = (rank + axis) // to make axe positive
        }


        newShape[axe] = inputShapes.sumOf { it[axe] } // concatenated dimension

        val tensorShape = newShape.clone()
        outputShape = tensorShape
        return tensorShape
    }

    override fun checkInputShapesOfInputOperands(input: List<Operand<Float>>) {
        require(input.size > 1) { "The number of input layers should be more than 1." }

        val firstInputShape = TensorShape(input[0].asOutput().shape())

        for (layer in input) {
            val tensorShape = TensorShape(
                layer.asOutput().shape()
            )
            require(
                firstInputShape.almostEqual(tensorShape, except = axis)
            ) {
                "A Concatenate layer requires inputs with matching shapes except for the concat axis. " +
                        "But shapes are the following: shape of first input is $firstInputShape and shape of layer $layer is $tensorShape."
            }
        }
    }

    override fun mergeFunction(input: List<Operand<Float>>, tf: Ops): Operand<Float> {
        return tf.concat(input, tf.constant(axis))
    }
}
