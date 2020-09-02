package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

private const val ACCUMULATOR = "gradient_accumulator"
private const val SQUARED_ACCUMULATOR = "gradient_squared_accumulator"

class AdaGradDA(
    private val learningRate: Float = 0.1f,
    private val initialAccumulatorValue: Float = 0.01f,
    private val l1Strength: Float = 0.01f,
    private val l2Strength: Float = 0.01f,
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private lateinit var learningRateConst: Constant<Float>
    private lateinit var l1StrengthConst: Constant<Float>
    private lateinit var l2StrengthConst: Constant<Float>
    private lateinit var globalStep: Variable<Long>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()
        learningRateConst = tf.constant(learningRate, getDType())
        l1StrengthConst = tf.constant(l1Strength, getDType())
        l2StrengthConst = tf.constant(l2Strength, getDType())

        val globalStepInitFinish1 = tf.assignAdd(globalStep, tf.constant(1L))
        graph.addOptimizerVariableAssignAddInitializer(globalStepInitFinish1)

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val gradSlot: Variable<Float> = getSlot(varName, ACCUMULATOR)
            val gradSquaredSlot: Variable<Float> = getSlot(varName, SQUARED_ACCUMULATOR)

            targets.add(
                tf.train.applyAdagradDa(
                    variable, gradSlot, gradSquaredSlot,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    learningRateConst,
                    l1StrengthConst,
                    l2StrengthConst,
                    globalStep
                )
            )
        }

        val globalStepInitFinish = tf.assignAdd(globalStep, tf.constant(1L))
        graph.addOptimizerVariableAssignAddInitializer(globalStepInitFinish)

        return targets
    }

    private fun createAdaGradDASlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val initializer: Operand<Float> = tf
            .fill(tf.shape(v), tf.dtypes.cast(tf.constant(0.0f), getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, initializer)
        val sqInitializer: Operand<Float> = tf.fill(
            tf.shape(v),
            tf.dtypes.cast(tf.constant(initialAccumulatorValue, getDType()), getDType())
        )
        createSlot(graph, tf, v.asOutput(), SQUARED_ACCUMULATOR, sqInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaGradDASlot(graph, tf, v.asOutput())
        }
        globalStep = tf.withName("adagrad-da-global-step").variable(Shape.scalar(), getLongType())
        val globalStepInit: Assign<Long> = tf.assign(globalStep, tf.constant(0L, getLongType()))
        graph.addOptimizerVariableInitializer(globalStepInit)
    }

    override fun getOptimizerName(): String {
        return "AdaGradDA"
    }
}

fun getLongType(): Class<Long> {
    return Long::class.javaObjectType
}