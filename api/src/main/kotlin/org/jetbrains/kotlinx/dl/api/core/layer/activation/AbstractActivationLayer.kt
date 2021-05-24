/*
 * Copyright 2021 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.activation

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public abstract class AbstractActivationLayer(name: String) : Layer(name) {

    init {
        this.isTrainable = false
    }

    public abstract fun forward(
        tf: Ops,
        input: Operand<Float>
    ): Operand<Float>

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> = forward(tf, input)

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape): Unit = Unit

    override fun computeOutputShape(inputShape: Shape): Shape {
        this.outputShape = TensorShape(inputShape)
        return inputShape
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = true

    override val paramCount: Int get() = 0
}
