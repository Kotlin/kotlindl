package tf_api.keras.layers

import examples.PADDING_TYPE
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

class AvgPool<T : Number>(private val poolSize: IntArray,
                          private val strides: IntArray) : Layer<T>() {
    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Empty method
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return inputShape
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