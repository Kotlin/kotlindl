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
 *
 */
public class Rescaling(
    public val scale: Double = 1.0,
    public val offset: Double = 0.0,
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
