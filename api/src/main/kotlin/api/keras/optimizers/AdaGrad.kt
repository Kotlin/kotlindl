package api.keras.optimizers

import api.KGraph
import api.defaultInitializerOpName
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

private const val ACCUMULATOR = "accumulator"

class AdaGrad(
    private val learningRate: Float = 0.1f,
    private val initialAccumulatorValue: Float = 0.01f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var initialAccumulatorValueConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        initialAccumulatorValueConstant = tf.constant(initialAccumulatorValue, getDType())
        learningRateConst = tf.constant(learningRate, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val slot: Variable<Float> = getSlot(varName, ACCUMULATOR)

            targets.add(
                tf.train.applyAdagrad(
                    variable,
                    slot,
                    learningRateConst,
                    clipGradient.clipGradient(tf, gradients.dy(i))
                )
            )
        }
        return targets
    }

    private fun createAdaGradSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))

        val initializer: Operand<Float> = tf.withName(accumInitializerName)
            .fill(tf.shape(v), tf.constant(initialAccumulatorValue))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, initializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaGradSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Adagrad"
    }
}