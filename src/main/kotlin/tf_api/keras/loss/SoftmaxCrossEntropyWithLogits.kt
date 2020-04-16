package tf_api.keras.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

private const val TRAINING_LOSS = "training_loss"

class SoftmaxCrossEntropyWithLogits<T : Number> : LossFunction<T> {
    override fun getTFOperand(tf: Ops, actual: Operand<T>, labels: Operand<T>): Operand<T> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(actual, labels)

        return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    }
}