package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

private const val ACCUMULATOR = "accumulator"

class AdaGrad<T : Number>(
    private val learningRate: Float = 0.1f,
    private val initialAccumulatorValue: Float = 0.01f
) : Optimizer<T>() {
    private lateinit var initialAccumulatorValueConstant: Constant<T>
    private lateinit var learningRateConst: Constant<T>

    override fun applyGradients(
        graph: KGraph<T>,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        initialAccumulatorValueConstant = tf.constant(initialAccumulatorValue, getDType())
        learningRateConst = tf.constant(learningRate, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val slot: Variable<T> = getSlot(varName, ACCUMULATOR)

            targets.add(
                tf.train.applyAdagrad(
                    variable, slot,
                    learningRateConst,
                    gradients.dy(i)
                )
            )

        }
        return targets
    }

    private fun createAdaGradSlot(graph: KGraph<T>, tf: Ops, v: Output<out T>) {
        val initializer: Operand<T> = tf.fill(
            tf.shape(v),
            tf.dtypes.cast(tf.constant(initialAccumulatorValue), getDType())
        )
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, initializer)
    }

    override fun createSlots(graph: KGraph<T>, tf: Ops, variables: List<Output<out T>>) {
        for (v in variables) {
            createAdaGradSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Adagrad"
    }
}