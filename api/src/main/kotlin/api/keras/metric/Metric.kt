package api.keras.metric

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface Metric {
    fun apply(tf: Ops, output: Operand<Float>, label: Operand<Float>, dtype: Class<Float>): Operand<Float>
}