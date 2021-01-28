/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.merge

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class Add(public val mergedLayers: List<Layer>, name: String = "") : Layer(name) {
    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {

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
        return tf.math.add(input, input)
    }


    override val weights: List<Array<*>>
        get() = TODO("Not yet implemented")
    override val hasActivation: Boolean
        get() = TODO("Not yet implemented")
    override val paramCount: Int
        get() = TODO("Not yet implemented")
}
