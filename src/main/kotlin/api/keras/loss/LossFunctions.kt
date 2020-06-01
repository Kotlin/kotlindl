package api.keras.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable


private const val TRAINING_LOSS = "training_loss"

enum class LossFunctions {
    SPARSE_CATEGORICAL_CROSS_ENTROPY,
    SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
    ABSOLUTE_DIFFERENCE,
    HINGE_LOSS,
    HUBER_LOSS,
    LOG_LOSS;

    companion object {
        fun <T : Number> convert(lossFunctionType: LossFunctions): LossFunction<T> {
            return when (lossFunctionType) {
                SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits()
                ABSOLUTE_DIFFERENCE -> AbsoluteDifference()
                SPARSE_CATEGORICAL_CROSS_ENTROPY -> TODO()
                HINGE_LOSS -> HingeLoss()
                HUBER_LOSS -> TODO()
                LOG_LOSS -> LogLoss()
            }
        }
    }
}

class SoftmaxCrossEntropyWithLogits<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(actual, labels)

        return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    }
}

class AbsoluteDifference<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        val losses = tf.math.abs(tf.math.sub(actual, labels))

        return tf.withName(TRAINING_LOSS).math.mean(tf.math.mean(losses, tf.constant(0)), tf.constant(0))
    }
}

class HingeLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {

        // We first need to convert binary labels to -1/1 labels (as floats).
        val allOnes: Variable<T> = tf.variable(labels.asOutput().shape(), dtype)
        // TODO: add assign operators
        val labelsShifted = tf.math.sub(tf.math.mul(tf.constant(2f) as Operand<T>, labels), allOnes)


        return tf.withName(TRAINING_LOSS).math.mean(
            tf.math.mean(
                tf.nn.relu(
                    tf.math.sub(allOnes, tf.math.mul(labelsShifted, actual))
                ), tf.constant(0)
            )
            , tf.constant(0)
        )
    }
}

class LogLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        val epsilon = 1e-5f

        val oneOp = tf.constant(1.0f) as Operand<T>
        val minusOneOp = tf.constant(-1.0f) as Operand<T>
        val epsilonOp = tf.constant(epsilon) as Operand<T>

        val right = tf.math.mul(labels, tf.math.log(tf.math.add(actual, epsilonOp)))
        val left =
            tf.math.mul(tf.math.log(tf.math.add(tf.math.sub(oneOp, actual), epsilonOp)), tf.math.sub(oneOp, labels))

        val sum = tf.math.add(right, left)
        return tf.withName(TRAINING_LOSS).math.mean(
            tf.reduceSum(tf.math.mul(minusOneOp, sum), tf.constant(0)),
            tf.constant(0)
        )
    }
}