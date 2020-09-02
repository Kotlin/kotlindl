package api.keras.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface LossFunction {
    fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float>
}
