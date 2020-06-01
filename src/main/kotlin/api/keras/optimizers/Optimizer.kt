package api.keras.optimizers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable

abstract class Optimizer<T : Number> {
    private lateinit var dtype: Class<T>

    fun prepareTargets(
        tf: Ops,
        loss: Operand<T>,
        weights: List<Variable<T>>,
        epochNumber: Int
    ): List<Operand<T>> {
        val gradients: Gradients = computeGradients(tf, loss, weights)
        return applyGradients(tf, weights, gradients, epochNumber)
    }

    protected abstract fun applyGradients(
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients,
        epochNumber: Int
    ): List<Operand<T>>

    private fun computeGradients(
        tf: Ops,
        loss: Operand<T>,
        weights: List<Variable<T>>
    ): Gradients {
        return tf.gradients(loss, weights)
    }

    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }
}