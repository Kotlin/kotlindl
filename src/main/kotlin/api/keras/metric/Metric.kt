package api.keras.metric

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface Metric<T : Number> {
    fun apply(tf: Ops, output: Operand<T>, label: Operand<T>, dtype: Class<T>): Operand<T>
}