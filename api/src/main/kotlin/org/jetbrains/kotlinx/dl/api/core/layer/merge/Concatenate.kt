/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.op.Ops

// TODO: for inception it should work with more than 2 inputs (4 for example) need to write more universal formulas here, for this layer and add
public class Concatenate(
    public var axis: Int = 3,
    name: String = ""
) : AbstractMerge("ConcatenateLayer", name) {
    override fun computeOutputShapeFromInboundLayers(): TensorShape {
        val inputShapes = mutableListOf<TensorShape>()
        inboundLayers.forEach { inboundLayer -> inputShapes.add(inboundLayer.outputShape) }
        val newShape = inputShapes[0].clone()

        var axe = axis
        if (axis == -1) { // TODO: I don't know how to handle this case correctly, it influences on nasmobilemodel
            val rank: Int = inputShapes[0].rank()
            axe = (rank + axis) // to make axe positive
        }


        newShape[axe] = inputShapes.map { it[axe] }.sum() // concatenated dimension

        val tensorShape = newShape.clone()
        outputShape = tensorShape
        return tensorShape
    }

    override fun checkInputShapesOfInputOperands(input: List<Operand<Float>>) {
// TODO: check (all shapes has the equal dimension) and same size on all dims except axis dimension
        // take shape from first input and replace axis dimension
    }

    override fun mergeFunction(input: List<Operand<Float>>, tf: Ops): Operand<Float> {
        return tf.concat(input, tf.constant(axis))
    }
}
