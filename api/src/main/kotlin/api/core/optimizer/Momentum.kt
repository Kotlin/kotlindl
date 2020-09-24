package api.core.optimizer

import api.core.KGraph
import api.core.util.defaultInitializerOpName
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyMomentum
import java.util.*

private const val MOMENTUM = "momentum"

/**
 * Improved version of [SGD] optimizer.
 *
 * @property [learningRate] Float >= 0. Initial learning rate.
 * @property [momentum] Float >= 0. Parameter that accelerates SGD in the relevant direction and dampens oscillations.
 * @property [useNesterov] If true, applies Nesterov momentum.
 */
class Momentum(
    private val learningRate: Float = 0.001f,
    private val momentum: Float = 0.99f,
    private val useNesterov: Boolean = true,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var momentumConst: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        learningRateConst = tf.constant(learningRate)
        momentumConst = tf.constant(momentum)

        for (i in weights.indices) {
            val variable = weights[i]

            val slot = getSlot(variable.ref().op().name(), MOMENTUM)

            targets.add(
                tf.train.applyMomentum(
                    variable,
                    slot,
                    learningRateConst,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    momentumConst,
                    ApplyMomentum.useNesterov(useNesterov)
                )
            )
        }
        return targets
    }

    private fun createMomentumSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val momentumInitializerName = defaultInitializerOpName(createName(v, MOMENTUM))
        val initializer: Operand<Float> = tf.withName(momentumInitializerName)
            .fill(tf.shape(v), tf.constant(0.0f))
        createSlot(graph, tf, v.asOutput(), MOMENTUM, initializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createMomentumSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Momentum"
    }
}