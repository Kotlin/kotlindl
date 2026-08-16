/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.ndarray.Shape
import org.tensorflow.op.Ops
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32

/**
 * Abstract Zero Padding layer used as the base layer for all the ZeroPadding layers.
 */
public abstract class AbstractZeroPadding(
    name: String
) : Layer(name) {
    override val hasActivation: Boolean get() = false

    override fun build(
        tf: Ops,
        input: Operand<TFloat32>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> {
        val inputShape = input.asOutput().shape()
        val paddingOperand = tf.constant(paddingArrayToTfFormat(inputShape))
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    /**
     * This function helps in computing the padding operand i.e. normalizing the padding array
     * into a tensorflow format. This method will then be called in [build] method that will be
     * further passed to tf.pad().
     */
    protected abstract fun paddingArrayToTfFormat(inputShape: Shape): Array<IntArray>
}
