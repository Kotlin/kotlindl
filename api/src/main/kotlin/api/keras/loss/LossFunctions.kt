package api.keras.loss

import api.TRAINING_LOSS
import api.getDType
import api.keras.Kmean
import org.tensorflow.Operand
import org.tensorflow.op.Ops

enum class LossFunctions {
    SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
    ABSOLUTE_DIFFERENCE,
    HINGE_LOSS,
    HUBER_LOSS,
    LOG_LOSS,
    MAE,
    MSE,
    RMSE;

    companion object {
        fun convert(lossFunctionType: LossFunctions): LossFunction {
            return when (lossFunctionType) {
                SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits()
                ABSOLUTE_DIFFERENCE -> AbsoluteDifference()
                HINGE_LOSS -> HingeLoss()
                HUBER_LOSS -> HuberLoss(0.5f)
                LOG_LOSS -> LogLoss()
                MAE -> MAE()
                MSE -> MSE()
                RMSE -> RMSE()
            }
        }
    }
}

class SoftmaxCrossEntropyWithLogits : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(actual, labels)

        return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    }
}

class AbsoluteDifference : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val losses = tf.math.abs(tf.math.sub(actual, labels))

        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(losses, tf.constant(-1)), tf.constant(0))
    }
}

class MAE : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val absoluteErrors = tf.math.abs(tf.math.sub(actual, labels))

        /*return tf.withName(TRAINING_LOSS).math.mean(absoluteErrors, tf.constant(0))*/
        /*return tf.withName(TRAINING_LOSS)
            .identity(Kmean(tf, absoluteErrors, tf.constant(-1)))*/

        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(absoluteErrors, tf.constant(-1)), tf.constant(0))
    }
}

class MSE : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val squaredError = tf.math.squaredDifference(actual, labels)
        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(squaredError, tf.constant(-1)), tf.constant(0))
    }
}

class RMSE : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val rootSquaredError = tf.math.sqrt(tf.math.squaredDifference(actual, labels))
        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(rootSquaredError, tf.constant(-1)), tf.constant(0))
    }
}

class HingeLoss : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        // We first need to convert binary labels to -1/1 labels (as floats).
        val labelsShifted = tf.math.sub(tf.math.mul(tf.constant(2f) as Operand<Float>, labels), labels)

        return tf.withName(TRAINING_LOSS).reduceSum(
            tf.math.mean(
                tf.nn.relu(
                    tf.math.sub(labels, tf.math.mul(labelsShifted, actual))
                ), tf.constant(-1)
            ), tf.constant(0)
        )
    }
}

// TODO: Huber is close to MAE/MSE via delta parameter: https://www.machinecurve.com/index.php/2019/10/12/using-huber-loss-in-keras/
class HuberLoss(val delta: Float) : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val error = tf.math.sub(actual, labels)

        val deltaConst: Operand<Float> =
            tf.dtypes.cast(tf.constant(delta), getDType()) // to actual.asOutput().dataType() in TF 2.x
        val point5: Operand<Float> = tf.dtypes.cast(tf.constant(0.5), getDType())

        val abs_error: Operand<Float> = tf.math.abs(error)
        val quadratic: Operand<Float> = tf.math.minimum(abs_error, deltaConst)
        val linear: Operand<Float> = tf.math.sub(abs_error, quadratic)

        val q2Point5: Operand<Float> =
            tf.math.mul(point5, tf.math.mul(quadratic, quadratic))

        val deltaLinear: Operand<Float> = tf.math.mul(deltaConst, linear)
        val loss: Operand<Float> = tf.math.add(q2Point5, deltaLinear)

        val result: Operand<Float> = Kmean(tf, loss, tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(tf.reduceSum(result, tf.constant(0)))
    }
}

class LogLoss : LossFunction {
    override fun apply(tf: Ops, actual: Operand<Float>, labels: Operand<Float>, dtype: Class<Float>): Operand<Float> {
        val epsilon = 1e-5f

        val oneOp = tf.constant(1.0f) as Operand<Float>
        val minusOneOp = tf.constant(-1.0f) as Operand<Float>
        val epsilonOp = tf.constant(epsilon) as Operand<Float>

        val right = tf.math.mul(labels, tf.math.log(tf.math.add(actual, epsilonOp)))
        val left =
            tf.math.mul(tf.math.log(tf.math.add(tf.math.sub(oneOp, actual), epsilonOp)), tf.math.sub(oneOp, labels))

        val sum = tf.math.add(right, left)
        return tf.withName(TRAINING_LOSS).reduceSum(
            tf.math.mean(tf.math.mul(minusOneOp, sum), tf.constant(-1)),
            tf.constant(0)
        )
    }
}