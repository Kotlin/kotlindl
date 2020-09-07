package api.keras.optimizers

import api.KGraph
import api.defaultAssignOpName
import api.defaultInitializerOpName
import api.defaultOptimizerVariableName
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

private val GLOBAL_STEP = defaultOptimizerVariableName("adagrad-da-global-step")
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
    private lateinit var globalStep: Variable<Float>

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

        for (i in weights.indices) {
            val variable = weights[i]
            val varName = variable.ref().op().name()

            val gradSlot: Variable<Float> = getSlot(varName, ACCUMULATOR)
            val gradSquaredSlot: Variable<Float> = getSlot(varName, SQUARED_ACCUMULATOR)

            targets.add(
                tf.train.applyAdagradDa(
                    variable,
                    gradSlot,
                    gradSquaredSlot,
                    clipGradient.clipGradient(tf, gradients.dy(i)),
                    learningRateConst,
                    l1StrengthConst,
                    l2StrengthConst,
                    tf.dtypes.cast(globalStep, Long::class.javaObjectType)
                )
            )
        }

        val globalStepInitFinish = tf.assignAdd(globalStep, tf.constant(1.0f))
        graph.addOptimizerVariableAssignAddInitializer(globalStepInitFinish)
        graph.addOptimizerVariable(globalStep)
        return targets
    }

    private fun createAdaGradDASlot(graph: KGraph, tf: Ops, v: Output<Float>) {
        val accumulatorInitializerName = defaultInitializerOpName(createName(v, ACCUMULATOR))
        val accumInitializer: Operand<Float> = tf.withName(accumulatorInitializerName)
            .fill(tf.shape(v), tf.constant(0.0f))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumInitializer)

        val squareAccumInitializerName = defaultInitializerOpName(createName(v, SQUARED_ACCUMULATOR))
        val sqInitializer: Operand<Float> = tf.withName(squareAccumInitializerName)
            .fill(tf.shape(v), tf.constant(initialAccumulatorValue))

        createSlot(graph, tf, v.asOutput(), SQUARED_ACCUMULATOR, sqInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<Float>>) {
        for (v in variables) {
            createAdaGradDASlot(graph, tf, v.asOutput())
        }
        globalStep = tf.withName(GLOBAL_STEP).variable(Shape.scalar(), getDType())
        val globalStepAssignName = defaultAssignOpName(GLOBAL_STEP)
        val globalStepInit: Assign<*> = tf.withName(globalStepAssignName)
            .assign(globalStep, tf.withName(defaultInitializerOpName(GLOBAL_STEP)).constant(0.0f))
        graph.addOptimizerVariableInitializer(globalStepInit)
    }

    override fun getOptimizerName(): String {
        return "AdaGradDA"
    }
}
