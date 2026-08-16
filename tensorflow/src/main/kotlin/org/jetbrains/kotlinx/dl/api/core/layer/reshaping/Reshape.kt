/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.types.TBool
import org.tensorflow.types.TFloat32
import org.tensorflow.types.TInt32

/**
 * Layer that reshapes inputs into the given shape.
 *
 * Input shape: `(batch_size,) + any shape`.
 *
 * Output shape: `(batch_size,) + target_shape`.
 *
 * @property [targetShape] Target shape. List of integers, does not include the samples dimension (batch size)
 * @property [name] Custom layer name.
 * @constructor Creates [Reshape] object.
 *
 * @since 0.2
 */
public class Reshape(
    public val targetShape: List<Int>,
    name: String = ""
) : Layer(name) {
    private lateinit var units: Constant<TInt32>

    override fun build(
        tf: Ops,
        input: Operand<TFloat32>,
        isTraining: Operand<TBool>,
        numberOfLosses: Operand<TFloat32>?
    ): Operand<TFloat32> {
        units = tf.constant(IntArray(targetShape.size + 1) {
            if (it == 0) -1 else targetShape[it - 1]
        })
        return tf.reshape(input, units)
    }

    override fun toString(): String {
        return "Reshape(name = $name, targetShape=$targetShape, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
