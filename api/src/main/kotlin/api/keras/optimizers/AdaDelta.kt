package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

private const val ACCUMULATOR = "accum"
private const val ACCUMULATOR_UPDATE = "accum_update"

class AdaDelta<T : Number>(
    private val learningRate: Float = 0.1f,
    private val rho: Float = 0.95f,
    private val epsilon: Float = 1e-8f
) : Optimizer<T>() {
    private lateinit var epsilonConstant: Constant<T>
    private lateinit var learningRateConst: Constant<T>
    private lateinit var rhoConst: Constant<T>

    override fun applyGradients(
        graph: KGraph<T>,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        rhoConst = tf.constant(rho, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val accumSlot: Variable<T> = getSlot(varName, ACCUMULATOR)
            val accumUpdateSlot: Variable<T> = getSlot(varName, ACCUMULATOR_UPDATE)

            targets.add(
                tf.train.applyAdadelta(
                    variable, accumSlot, accumUpdateSlot,
                    learningRateConst,
                    rhoConst,
                    epsilonConstant,
                    gradients.dy(i)
                )
            )

        }
        return targets
    }

    private fun createAdaDeltaSlot(graph: KGraph<T>, tf: Ops, v: Output<out T>) {
        val accumulatorInitializer = tf
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumulatorInitializer)
        val updateInitializer: Operand<T> = tf
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR_UPDATE, updateInitializer)
    }

    override fun createSlots(graph: KGraph<T>, tf: Ops, variables: List<Output<out T>>) {
        for (v in variables) {
            createAdaDeltaSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Adadelta"
    }
}