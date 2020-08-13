package api.keras.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface LossFunction<T : Number> {
    fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T>
}
