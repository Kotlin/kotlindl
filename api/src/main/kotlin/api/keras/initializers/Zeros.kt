package api.keras.initializers

import api.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

class Zeros : Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).zeros(shape, getDType())
    }
}
