/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public abstract class AbstractCropping(
    public val croppingInternal: Array<IntArray>,
    name: String,
) : Layer(name) {
    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        return computeOutputShape(inputShape)
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        return crop(tf, input)
    }

    protected abstract fun crop(tf: Ops, input: Operand<Float>): Operand<Float>

    protected abstract fun computeCroppedShape(inputShape: Shape): Shape
}
