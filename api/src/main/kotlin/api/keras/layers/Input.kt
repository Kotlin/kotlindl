package api.keras.layers

import api.core.KGraph
import api.keras.util.DATA_PLACEHOLDER
import api.keras.util.getDType
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder

class Input(vararg dims: Long, name: String = "") : Layer(name) {
    lateinit var input: Placeholder<Float>

    val packedDims: LongArray = dims

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    fun defineVariables(tf: Ops) {
        input = tf.withName(DATA_PLACEHOLDER).placeholder(
            getDType(),
            Placeholder.shape(Shape.make(-1L, *packedDims))
        )
    }

    fun computeOutputShape(): Shape {
        return input.asOutput().shape()
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        return input
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
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
        return "Input(shape=${packedDims.contentToString()})"
    }
}