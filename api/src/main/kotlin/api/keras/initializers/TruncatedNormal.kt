package api.keras.initializers

import api.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.random.TruncatedNormal

class TruncatedNormal(private val seed: Long) :
    Initializer() {
    override fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float> {
        return tf.withName(name).random.truncatedNormal(
            shape,
            getDType(),
            TruncatedNormal.seed(seed)
        )
    }

    override fun toString(): String {
        return "TruncatedNormal(seed=$seed)"
    }
}