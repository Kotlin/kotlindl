package api.keras.layers.twodim

import api.KGraph
import api.keras.layers.Layer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

private const val PADDING_TYPE = "SAME"

class AvgPool2D<T : Number>(
    private val poolSize: IntArray,
    private val strides: IntArray
) : Layer<T>() {
    override fun defineVariables(tf: Ops, kGraph: KGraph<T>, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var rows = inputShape.size(1)
        var cols = inputShape.size(2)
        rows = poolOutputLength(
            rows, poolSize[1],
            strides[1]
        )
        cols = poolOutputLength(
            cols, poolSize[2],
            strides[2]
        )

        return Shape.make(inputShape.size(0), rows, cols, inputShape.size(3))
    }

    // TODO: for different paddings use this function https://github.com/keras-team/keras/blob/7a39b6c62d43c25472b2c2476bd2a8983ae4f682/keras/utils/conv_utils.py#L85
    private fun poolOutputLength(inputLength: Long, filterSize: Int, stride: Int): Long {
        return ((inputLength + stride - 1).toFloat() / stride).toLong()
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {

        // data conversion due to different signatures of nn.avgPool and nn.maxPool
        val poolSizeLongList: MutableList<Long> = mutableListOf()
        poolSize.forEach {
            poolSizeLongList.add(it.toLong())
        }

        val stridesLongList: MutableList<Long> = mutableListOf()
        strides.forEach {
            stridesLongList.add(it.toLong())
        }

        return tf.nn.avgPool(
            input,
            poolSizeLongList,
            stridesLongList,
            PADDING_TYPE // TODO: could it be changed?
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
}