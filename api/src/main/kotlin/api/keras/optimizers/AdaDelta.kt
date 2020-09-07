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

private const val ACCUMULATOR = "accum"
private const val ACCUMULATOR_UPDATE = "accum_update"

class AdaDelta(
    private val learningRate: Float = 0.1f,
    private val rho: Float = 0.95f,
    private val epsilon: Float = 1e-8f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var epsilonConstant: Constant<Float>
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var rhoConst: Constant<Float>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        rhoConst = tf.constant(rho, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val accumSlot: Variable<Float> = getSlot(varName, ACCUMULATOR)
            val accumUpdateSlot: Variable<Float> = getSlot(varName, ACCUMULATOR_UPDATE)

            targets.add(
                tf.train.applyAdadelta(
                    variable, accumSlot, accumUpdateSlot,
                    learningRateConst,
                    rhoConst,
                    epsilonConstant,
                    clipGradient.clipGradient(tf, gradients.dy(i))
                )
            )

        }
        return targets
    }

    private fun createAdaDeltaSlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))
        val accumulatorInitializer = tf.withName(accumInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumulatorInitializer)

        val accumUpdateInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR_UPDATE))
        val updateInitializer: Operand<Float> = tf.withName(accumUpdateInitializerName)
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR_UPDATE, updateInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaDeltaSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Adadelta"
    }
}