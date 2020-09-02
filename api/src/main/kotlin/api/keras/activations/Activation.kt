package api.keras.activations

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface Activation {
    fun apply(tf: Ops, features: Operand<Float>, name: String = ""): Operand<Float>
}