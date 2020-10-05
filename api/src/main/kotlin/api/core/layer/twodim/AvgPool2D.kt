package api.core.layer.twodim

import api.core.KGraph
import api.core.layer.Layer
import api.core.shape.convOutputLength
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

/**
 * Average pooling layer for 2D inputs (e.g. images).
 *
 * NOTE: Works with tensors which must have rank 4 (batch, height, width, channels).
 *
 * @property [poolSize] The size of the sliding window for each dimension of input tensor (pool batch, pool height, pool width, pool channels).
 * Usually, pool batch and pool channels are equal to 1.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * @property [padding] The padding method, either 'valid' or 'same' or 'full'.
 * @property [name] Custom layer name.
 * @constructor Creates [AvgPool2D] object.
 */
public class AvgPool2D(
    public val poolSize: IntArray = intArrayOf(1, 2, 2, 1),
    public val strides: IntArray = intArrayOf(1, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
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