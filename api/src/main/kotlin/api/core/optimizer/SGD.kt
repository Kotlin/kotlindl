package api.core.optimizer

import api.core.KGraph
import api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

/**
 * Stochastic gradient descent optimizer.
 *
 * NOTE: It's not an equivalent for keras.sgd, it's pure SGD with simple 'variable' update by subtracting 'alpha' * 'delta' from it.
 */
class SGD(
    clipGradient: ClipGradientAction = NoClipGradient()
) : Optimizer(clipGradient) {
    private var learningRate: Float = 0.2f

    constructor(learningRate: Float = 0.2f, clipGradient: ClipGradientAction = NoClipGradient()) : this() {
        this.learningRate = learningRate
    }

    override fun applyGradients(
        graph: KGraph,
        tf: Ops,
        weights: List<Variable<Float>>,
        gradients: Gradients
    ): List<Operand<Float>> {
        val targets: MutableList<Operand<Float>> =
            ArrayList()

        for (i in weights.indices) {
            targets.add(
                tf.train.applyGradientDescent(
                    weights[i],
                    tf.constant(learningRate, getDType()),
                    clipGradient.clipGradient(tf, gradients.dy(i))
                )
            )
        }

        return targets
    }

    override fun getOptimizerName(): String {
        return "SGD"
    }
}