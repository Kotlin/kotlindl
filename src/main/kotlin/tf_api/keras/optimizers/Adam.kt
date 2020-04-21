package tf_api.keras.optimizers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Gradients
import org.tensorflow.op.core.Variable
import java.util.*

// TODO: https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/training/adam.py#L32-L235
// TODO: copy the optimizer from here https://github.com/tensorflow/java/blob/master/tensorflow-training/src/main/java/org/tensorflow/training/optimizers/Adam.java
class Adam<T : Number>(
    private val beta1Power: Float, private val beta2Power: Float,
    private val learningRate: Float,
    private val beta1: Float,
    private val beta2: Float,
    private val epsilon: Float
) : Optimizer<T>() {
    override fun applyGradients(tf: Ops, weights: List<Variable<T>>, gradients: Gradients): List<Operand<T>> {
        val targets: MutableList<Operand<T>> =
            ArrayList()

        val m: Variable<T> = tf.variable(Shape.scalar(), getDType())
        val v: Variable<T> = tf.variable(Shape.scalar(), getDType())

        tf.assign(m, tf.constant(0f, getDType()))
        tf.assign(v, tf.constant(0f, getDType()))

        for (i in weights.indices) {
            targets.add(
                tf.train.applyAdam(
                    weights[i],
                    m,
                    v,
                    tf.constant(beta1Power, getDType()),
                    tf.constant(beta2Power, getDType()),
                    tf.constant(learningRate, getDType()),
                    tf.constant(beta1, getDType()),
                    tf.constant(beta2, getDType()),
                    tf.constant(epsilon, getDType()),
                    gradients.dy<T>(i)
                )
            )
        }

        return targets
    }
}