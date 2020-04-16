package tf_api.keras

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops

fun constArray(tf: Ops, vararg i: Int): Operand<Int> {
    return tf.constant(i)
}

fun shapeOperand(tf: Ops, shape: Shape): Operand<Int> {
    val shapeArray = IntArray(shape.numDimensions())
    for (i in shapeArray.indices) {
        shapeArray[i] = shape.size(i).toInt()
    }
    return tf.constant(shapeArray)
}