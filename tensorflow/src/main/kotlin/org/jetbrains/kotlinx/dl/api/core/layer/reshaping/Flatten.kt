/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.numElements
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import kotlin.math.abs

/**
 * Flattens the input. Does not affect the batch size.
 *
 * @property [name] Custom layer name.
 * @constructor Creates [Flatten] object.
 */
public class Flatten(name: String = "") : Layer(name) {
    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val inputShape = input.asOutput().shape()
        val amountOfNeuronsInFlattenLayer = (inputShape.numElements() / abs(inputShape.size(0))).toInt()
        val units = tf.constant(intArrayOf(-1, amountOfNeuronsInFlattenLayer))
        return tf.reshape(input, units)
    }

    override fun toString(): String {
        return "Flatten(name = $name, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
