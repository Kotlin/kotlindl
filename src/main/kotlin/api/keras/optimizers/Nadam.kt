package api.keras.optimizers

import api.KGraph
import org.tensorflow.Operand
import org.tensorflow.Output
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.Scope
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Constant
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import org.tensorflow.op.train.ApplyAdaMax
import java.util.*

private const val FIRST_MOMENT = "m"
private const val SECOND_MOMENT = "v"
private const val MOMENTUM = "momentum"

class Nadam<T : Number>(
    private val learningRate: Float = 0.001f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val epsilon: Float = 1e-07f,
    private val decayBase: Float = 0.96f,
    private val decay: Float = 0.004f
) : Optimizer<T>() {


    private lateinit var learningRateConst: Constant<T>
    private lateinit var betaOneConst: Constant<T>
    private lateinit var betaTwoConst: Constant<T>
    private lateinit var localStepConst: Constant<T>
    private lateinit var nextStepConst: Constant<T>
    private lateinit var decayBaseConst: Constant<T>
    private lateinit var decayConst: Constant<T>
    private lateinit var epsilonConstant: Constant<T>


    private lateinit var betaOnePower: Variable<T>
    private lateinit var betaTwoPower: Variable<T>
    private lateinit var momentum: Variable<T>

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        betaOneConst = tf.constant(beta1, getDType())
        betaTwoConst = tf.constant(beta2, getDType())
        learningRateConst = tf.constant(learningRate, getDType())
        epsilonConstant = tf.constant(epsilon, getDType())

        val scope = Scope(graph.tfGraph)

        for (i in weights.indices) {
            val firstMomentSlot: Variable<T> = getSlot(weights[i].ref().op().name(), FIRST_MOMENT)
            val secondMomentSlot: Variable<T> = getSlot(weights[i].ref().op().name(), SECOND_MOMENT)

            targets.add(
                ApplyAdaMax.create(
                    scope,
                    weights[i],
                    firstMomentSlot,
                    secondMomentSlot,
                    betaOnePower,
                    learningRateConst,
                    betaOneConst,
                    betaTwoConst,
                    epsilonConstant,
                    gradients.dy(i)
                )
            )
        }


        val betaOnePowerInit2 = tf.assign(betaOnePower, tf.math.mul(betaOnePower, betaOneConst))
        val betaTwoPowerInit2 = tf.assign(betaTwoPower, tf.math.mul(betaTwoPower, betaTwoConst))

        graph.optimizerInitializers += betaOnePowerInit2
        graph.optimizerInitializers += betaTwoPowerInit2

        return targets
    }

    private fun createNadamSlot(graph: KGraph, tf: Ops, v: Output<out T>) {
        val firstMomentInitializer = tf.fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), FIRST_MOMENT, firstMomentInitializer)

        val secondMomentInitializer = tf.fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), SECOND_MOMENT, secondMomentInitializer)

        val momentumInitializer = tf.fill(tf.shape(v), tf.constant(0.0f, getDType()))
        createSlot(graph, tf, v.asOutput(), MOMENTUM, momentumInitializer)
    }

    override fun createSlots(graph: KGraph, tf: Ops, variables: List<Output<out T>>) {
        for (v in variables) {
            createNadamSlot(graph, tf, v.asOutput())
        }
        betaOnePower = tf.withName("beta1_power").variable(Shape.scalar(), getDType())
        val betaOnePowerInit: Assign<T> = tf
            .assign(betaOnePower, tf.constant(beta1, getDType()))
        graph.optimizerInitializers += betaOnePowerInit

        betaTwoPower = tf.withName("beta2_power").variable(Shape.scalar(), getDType())
        val betaTwoPowerInit: Assign<T> = tf
            .assign(betaTwoPower, tf.constant(beta2, getDType()))
        graph.optimizerInitializers += betaTwoPowerInit

        momentum = tf.withName(MOMENTUM).variable(Shape.scalar(), getDType())
        val momentumInit: Assign<T> = tf
            .assign(momentum, tf.constant(beta2, getDType()))
        graph.optimizerInitializers += momentumInit
    }

    override fun getOptimizerName(): String {
        return "Nadam"
    }
}