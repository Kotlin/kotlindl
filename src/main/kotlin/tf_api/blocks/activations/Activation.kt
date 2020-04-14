package tf_api.blocks.activations

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface Activation<T : Number> {
    fun apply(tf: Ops, features: Operand<T>): Operand<T>
}