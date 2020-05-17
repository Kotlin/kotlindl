package tf_api.keras.optimizers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

class SGD<T : Number>(private val learningRateSchedule: Map<Int, Float>) : Optimizer<T>() {
    private var learningRate: Float = 0.2f

    constructor(learningRate: Float) : this(mapOf()) {
        this.learningRate = learningRate
    }

    override fun applyGradients(
        tf: Ops,
        weights: List<Variable<T>>,
        gradients: Gradients,
        epochNumber: Int
    ): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        if (learningRateSchedule.isNotEmpty()) {
            val currentEpochLearningRate = learningRateSchedule[epochNumber]
            if (currentEpochLearningRate != null) {
                for (i in weights.indices) {
                    targets.add(
                        tf.train.applyGradientDescent(
                            weights[i],
                            tf.constant(currentEpochLearningRate, getDType()),
                            gradients.dy<T>(i)
                        )
                    )
                }
            } else {
                throw Exception("No schedule for the epoch: $epochNumber")
            }
        } else {
            for (i in weights.indices) {
                targets.add(
                    tf.train.applyGradientDescent(
                        weights[i],
                        tf.constant(learningRate, getDType()),
                        gradients.dy<T>(i)
                    )
                )
            }
        }



        return targets
    }
}