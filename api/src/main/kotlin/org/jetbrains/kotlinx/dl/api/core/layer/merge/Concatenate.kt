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

// TODO: for inception it should work with more than 2 inputs (4 for example) need to write more universal formulas here, for this layer and add
public class Concatenate(
    public var axis: Int = 3,
    name: String = ""
) : Layer(name) {
    public val mergedLayers: List<Layer> = emptyList()

    init {
        inboundLayers = mergedLayers as MutableList<Layer>
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {

    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        val inputShapes = mutableListOf<LongArray>()
        inboundLayers.forEach { inboundLayer -> inputShapes.add(inboundLayer.outputShape) }
        val newShape = inputShapes[0].clone()

        var axe = axis
        if (axis == -1) { // TODO: I don't know how to handle this case correctly, it influences on nasmobilemodel
            val rank: Int = inputShapes[0].size
            axe = rank + axis // to make axe positive
            /*if (rank != 0) {
                axe %= rank
            } else {
                axe = 0;
            }*/


        }
        newShape[axe] = inputShapes.map { it[axe] }.sum() // concatenated dimension
        // TODO: check (all shapes has the equal dimension) and same size on all dims except axis dimension
        // take shape from first input and replace axis dimension


        val tensorShape = TensorShape(newShape)
        outputShape = tensorShape.dims()
        return tensorShape.toShape()
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // TODO: this call should be banned or merged with the following method forward()
        return input
    }

    override fun forward(
        tf: Ops,
        input: List<Operand<Float>>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return tf.withName("CONCAT_LAYER").concat(input, tf.constant(axis))
    }

    override val weights: List<Array<*>>
        get() = emptyList()
    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
}
