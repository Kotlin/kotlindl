/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class ActivationLayer(
    public val activation: Activations = Activations.Relu,
    name: String = ""
) : Layer(name) {

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
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
        return Activations.convert(activation).apply(tf, input, name)
    }

    override fun getWeights(): List<Array<*>> {
        return emptyList()
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return 0
    }

    override fun toString(): String {
        return "ActivationLayer(activation=$activation)"
    }
}
