/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package api.core.layer

import api.core.KGraph
import api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.random.RandomUniform

/**
 * Applies Dropout to the input.
 *
 * Dropout consists in randomly setting a fraction `rate` of input units to 0
 * at each update during training time, which helps prevent overfitting.
 * The units that are kept are scaled by `1 / (1 - rate)`, so that their
 * sum is unchanged at training time and inference time.
 *
 * NOTE: Import and export for this layer is not supported yet.
 *
 * @property keepProbability The dropout rate, between 0 and 1. E.g. `rate=0.1` would drop out 10% of input units.
 * @property [name] Custom layer name.
 * @constructor Creates [Dropout] object.
 */
public class Dropout(
    private val keepProbability: Float = 0.1f,
    private val seed: Long = 12L,
    name: String = ""
) : Layer(name) {

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        val trainingFactor = tf.placeholderWithDefault(tf.constant(0.0f), Shape.scalar())

        val probability = tf.math.add(
            tf.math.mul(trainingFactor, tf.constant(keepProbability - 1.0f)),
            tf.constant(1.0f)
        )// When training

        val inputShape = input.asOutput().shape()
        val dims = mutableListOf<Long>()
        for (i in 1 until inputShape.numDimensions()) // skip first dimension
            dims.add(inputShape.size(i))

        val options = RandomUniform.seed(seed).seed2(seed + 1)
        val randomUniform = tf.random.randomUniform(tf.constant(dims.toLongArray()), getDType(), options)

        val mask = tf.math.floor(tf.math.add(randomUniform, probability as Operand<Float>))

        return tf.math.div(tf.math.mul(input, mask), probability)
    }

    override fun getWeights(): List<Array<*>> {
        return emptyList()
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return 0
    }

    override fun toString(): String {
        return "Dropout(keepProbability=$keepProbability)"
    }
}