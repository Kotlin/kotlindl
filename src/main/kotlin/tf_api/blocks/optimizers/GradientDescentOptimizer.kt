package tf_api.blocks.optimizers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

class GradientDescentOptimizer<T : Number>(private val learningRate: Float) : Optimizer<T>() {
    override fun applyGradients(tf: Ops, weights: List<Variable<T>>, gradients: Gradients): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        for (i in weights.indices) {
            targets.add(
                tf.train.applyGradientDescent(
                    weights[i],
                    tf.constant(learningRate, getDType()),
                    gradients.dy<T>(i)
                )
            )
        }

        return targets
    }
}