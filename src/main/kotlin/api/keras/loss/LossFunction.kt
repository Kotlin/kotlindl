package api.keras.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

interface LossFunction<T : Number> {
    fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T>
}
