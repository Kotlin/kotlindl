package api.keras.layers.twodim

import api.KGraph
import api.keras.layers.Layer
import api.keras.shape.convOutputLength
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class AvgPool2D(
    val poolSize: IntArray,
    val strides: IntArray,
    val padding: ConvPadding = ConvPadding.SAME,
    name: String = ""
) : Layer(name) {
    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = convOutputLength(
            rows, poolSize[1], padding,
            strides[1]
        )
        cols = convOutputLength(
            cols, poolSize[2], padding,
            strides[2]
        )

        return Shape.make(inputShape.size(0), rows, cols, inputShape.size(3))
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        // data conversion due to different signatures of nn.avgPool and nn.maxPool
        val poolSizeLongList: MutableList<Long> = mutableListOf()
        poolSize.forEach {
            poolSizeLongList.add(it.toLong())
        }

        val stridesLongList: MutableList<Long> = mutableListOf()
        strides.forEach {
            stridesLongList.add(it.toLong())
        }

        val tfPadding = when (padding) {
            ConvPadding.SAME -> "SAME"
            ConvPadding.VALID -> "VALID"
            ConvPadding.FULL -> "FULL"
        }

        return tf.nn.avgPool(
            input,
            poolSizeLongList,
            stridesLongList,
            tfPadding
        )
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
        return "AvgPool2D(poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding)"
    }
}