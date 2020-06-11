package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyMomentum
import java.util.*

const val MOMENTUM = "momentum"

class Momentum<T : Number>(
    private val learningRate: Float = 0.001f,
    private val momentum: Float = 0.01f,
    private val useNesterov: Boolean = true
) : Optimizer<T>() {

    private lateinit var momentumConst: Constant<T>
    private lateinit var learningRateConst: Constant<T>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        learningRateConst = tf.constant(learningRate, getDType())
        momentumConst = tf.constant(momentum, getDType())

        for (i in weights.indices) {
            val variable = weights[i]

            val slot = getSlot(variable.ref().op().name(), MOMENTUM)

            targets.add(
                tf.train.applyMomentum(
                    variable,
                    slot,
                    learningRateConst,
                    gradients.dy(i),
                    momentumConst,
                    ApplyMomentum.useNesterov(useNesterov)
                )
            )
        }
        return targets
    }

    private fun createMomentumSlot(graph: KGraph, tf: Ops, v: Output<out T>) {
        val initializer: Operand<T> = tf.fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), MOMENTUM, initializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<out T>>) {
        for (v in variables) {
            createMomentumSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Momentum"
    }
}