package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.util.TF
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

public class GlobalAvgPool3D(
    name: String = ""
):Layer(name) {
    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) { }

    override fun computeOutputShape(inputShape: Shape): Shape {
        //TODO support for dataFormat
        return Shape.make(inputShape.size(0), inputShape.size(4))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        //TODO support for dataFormat
        return TF.mean(tf, input, tf.constant(intArrayOf(1, 2, 3)))
    }

    override val weights: Map<String, Array<*>>
        get() = emptyMap()

    override val hasActivation: Boolean
        get() = false

    override val paramCount: Int
        get() = 0

    override fun toString(): String {
        return "GlobalAvgPool3D()"
    }
}
