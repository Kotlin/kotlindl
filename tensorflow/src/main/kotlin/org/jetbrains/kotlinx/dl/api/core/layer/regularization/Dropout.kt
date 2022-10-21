/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.regularization

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.NoGradients
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Applies Dropout to the input.
 *
 * The Dropout layer randomly sets input units to 0 with a frequency of `rate`
 * at each step during training time, which helps prevent overfitting.
 *
 * Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
 * all inputs is unchanged.
 *
 * NOTE: Export for this layer is not supported yet.
 * NOTE: This layer used for inference purposes only.
 *
 * @property [rate] A fraction of the input units to drop, should be between 0 and 1.
 * @property [seed] A seed for the random number generator.
 * @param    [name] Custom layer name.
 * @constructor Creates [Dropout] object.
 */
public class Dropout(
    public val rate: Float = 0.1f,
    public val seed: Long = 12L,
    name: String = ""
) : Layer(name), NoGradients {

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        /* if (isTraining) {
             val trainingFactor = tf.placeholderWithDefault(tf.constant(1.0f), Shape.scalar())

             val probability = tf.math.add(
                 tf.math.mul(trainingFactor, tf.constant(keepProbability - 1.0f)),
                 tf.constant(1.0f)
             ) // When training

             val inputShape = input.asOutput().shape()
             val dims = mutableListOf<Long>()
             for (i in 1 until inputShape.numDimensions()) // skip first dimension
                 dims.add(inputShape.size(i))

             val options = RandomUniform.seed(seed).seed2(seed + 1)
             val randomUniform = tf.random.randomUniform(tf.constant(dims.toLongArray()), getDType(), options)

             val mask = tf.math.floor(tf.math.add(randomUniform, probability as Operand<Float>))

             return tf.math.div(tf.math.mul(input, mask), probability)
         } else {*/
        return input
        /* }*/
    }

    override fun toString(): String {
        return "Dropout(name = $name, rate=$rate, seed=$seed, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
