package tf_api.keras.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant

class Flatten<T : Number> : Layer<T>() {
    private val NEED_TO_CALCULATE = 400
    private lateinit var units: Constant<Int>

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // NEED_TO_CALCULATE = should be calculated
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        var amoutOfElements = 0L
        for(i in 0 until inputShape.numDimensions())
            amoutOfElements += inputShape.size(i)

        return Shape.make(amoutOfElements)
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
       return tf.reshape(input, tf.constant(NEED_TO_CALCULATE))
    }
}