package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyFtrl
import java.util.*

private const val ACCUMULATOR = "gradient_accumulator"
private const val LINEAR_ACCUMULATOR = "linear_accumulator"


/**
 * ApplyFtrlV2 is supported for CPU only (Exception in thread "main" java.lang.IllegalArgumentException on GPU)
 */
class Ftrl<T : Number>(
    private val learningRate: Float = 0.001f,
    private val l1RegularizationStrength: Float = 0.0f,
    private val l2RegularizationStrength: Float = 0.0f,
    private val learningRatePower: Float = -0.5f,
    private val l2ShrinkageRegularizationStrength: Float = 0.0f
) : Optimizer<T>() {

    private var initialAccumulatorValue = 0.0f
    private lateinit var learningRatePowerConst: Constant<T>
    private lateinit var learningRateConst: Constant<T>
    private lateinit var l1RegularizationStrengthConst: Constant<T>
    private lateinit var l2RegularizationStrengthConst: Constant<T>
    private lateinit var l2ShrinkageRegularizationStrengthConst: Constant<T>

    override fun applyGradients(
        graph: KGraph<T>,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        l1RegularizationStrengthConst = tf.constant(l1RegularizationStrength, getDType())
        l2RegularizationStrengthConst = tf.constant(l2RegularizationStrength, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        l2ShrinkageRegularizationStrengthConst = tf.constant(l2ShrinkageRegularizationStrength, getDType())
        learningRatePowerConst = tf.constant(learningRatePower, getDType())

        for (i in weights.indices) {

            val variable = weights[i]
            val varName = variable.ref().op().name()

            val accumSlot: Variable<T> = getSlot(varName, ACCUMULATOR)
            val linearSlot: Variable<T> = getSlot(varName, LINEAR_ACCUMULATOR)
            val options = ApplyFtrl.useLocking(true)

            targets.add(
                tf.train.applyFtrl(
                    variable,
                    accumSlot,
                    linearSlot,
                    gradients.dy(i),
                    learningRateConst,
                    l1RegularizationStrengthConst,
                    l2RegularizationStrengthConst,
                    l2ShrinkageRegularizationStrengthConst,
                    learningRatePowerConst,
                    options
                )
            )
        }

        return targets
    }

    private fun createFtrlSlot(graph: KGraph<T>, tf: Ops, v: Output<out T>) {
        val accumInitializer = tf.fill(tf.shape(v), tf.constant(initialAccumulatorValue, getDType()))
        createSlot(graph, tf, v.asOutput(), ACCUMULATOR, accumInitializer)

        val linearAccumInitializer = tf.fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), LINEAR_ACCUMULATOR, linearAccumInitializer)
    }

    override fun createSlots(graph: KGraph<T>, tf: Ops, variables: List<Output<out T>>) {
        for (v in variables) {
            createFtrlSlot(graph, tf, v.asOutput())
        }
    }

    override fun getOptimizerName(): String {
        return "Ftrl"
    }
}