/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.preprocessing

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Feature-wise normalization of the data.

This layer will coerce its inputs into a distribution centered around
0 with standard deviation 1. It accomplishes this by precomputing the mean and
variance of the data, and calling (input-mean)/sqrt(var) at runtime.
 */
public class Normalization(
    /**
     * Integer or tuple of integers, the axis or axes that should be
    "kept". These axes are not be summed over when calculating the
    normalization statistics. By default the last axis, the `features` axis
    is kept and any `space` or `time` axes are summed. Each element in the
    the axes that are kept is normalized independently. If `axis` is set to
    'None', the layer will perform scalar normalization (dividing the input
    by a single scalar value). The `batch` axis, 0, is always summed over
    (`axis=0` is not allowed).
     */
    public val axis: Int = -1,

    /** The mean value(s) to use during normalization. */
    public val mean: Double = 0.0,

    /** The variance value(s) to use during normalization. */
    public val variance: Double = 0.0,
    name: String = ""
) : Layer(name) {

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        //left empty
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return input
    }

    override val weights: List<Array<*>> get() = emptyList()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0
}
