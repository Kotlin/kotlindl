package tf_api.blocks.layers

import examples.PADDING_TYPE
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class AvgPool<T : Number>(private val poolSize: IntArray,
                          private val strides: IntArray) : Layer<T>() {
    override fun addTFOperands(tf: Ops, inputShape: Shape) {
        TODO("Not yet implemented")
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        TODO("Not yet implemented")
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        return tf.nn.maxPool(
            input,
            tf.constant(poolSize),
            tf.constant(strides),
            PADDING_TYPE
        )
    }
}