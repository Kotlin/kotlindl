/*
 * Copyright 2021-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */
package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Permutes the dimensions of the input according to a given pattern.
 * Input shape: Arbitrary
 * Output shape: Same as input shape, but with the dimensions re-ordered according to the specified pattern.
 * @property [dims] array of integers.
 * @property [name] Custom layer name.
 * @constructor Creates [Permute] object.
 *
 * @since 0.3
 */
public class Permute(
    public val dims: IntArray,
    name: String = ""
) : Layer(name) {

    init {
        require(dims.sorted() == (1..dims.size).toList()) {
            "The values in dims should be consecutive and start from 1."
        }
    }

    override fun build(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val permArray = intArrayOf(0) + dims
        val perm = tf.constant(permArray)
        return tf.linalg.transpose(input, perm)
    }

    override fun toString(): String {
        return "Permute(name = $name, dims=${dims.contentToString()}, hasActivation=$hasActivation)"
    }

    override val hasActivation: Boolean get() = false
}
