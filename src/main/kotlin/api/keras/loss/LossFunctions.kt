package api.keras.loss

import api.TRAINING_LOSS
import org.tensorflow.Operand
import org.tensorflow.op.Ops

//https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/losses.ts
enum class LossFunctions {
    SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
    ABSOLUTE_DIFFERENCE,
    HINGE_LOSS,
    HUBER_LOSS,
    LOG_LOSS,
    MAE,
    MSE;

    companion object {
        fun <T : Number> convert(lossFunctionType: LossFunctions): LossFunction<T> {
            return when (lossFunctionType) {
                SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits()
                ABSOLUTE_DIFFERENCE -> AbsoluteDifference()
                HINGE_LOSS -> HingeLoss()
                HUBER_LOSS -> HuberLoss(0.1f)
                LOG_LOSS -> LogLoss()
                MAE -> MAE()
                MSE -> MSE()
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

class MAE<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        // tfc.mean(tfc.abs(tfc.sub(yPred, yTrue)), -1))
        val absoluteErrors = tf.math.abs(tf.math.sub(actual, labels))

        return tf.withName(TRAINING_LOSS).math.mean(tf.math.mean(absoluteErrors, tf.constant(-1)), tf.constant(-1))
        /*return tf.withName(TRAINING_LOSS).identity(Kmean(tf, tf.math.abs(tf.math.sub(actual, labels)), tf.constant(-1)));*/
    }
}

class MSE<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        val squaredError = tf.math.squaredDifference(actual, labels)
        return tf.withName(TRAINING_LOSS).math.mean(tf.math.mean(squaredError, tf.constant(-1)), tf.constant(-1))
    }
}

class HingeLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        val sub = tf.math.sub(tf.constant(1f) as Operand<T>, tf.math.mul(actual, labels))
        val maxResult = tf.math.maximum(tf.constant(0f) as Operand<T>, sub) as Operand<T>
        return tf.withName(TRAINING_LOSS).math.mean(tf.math.mean(maxResult, tf.constant(-1)), tf.constant(-1))
        /* // We first need to convert binary labels to -1/1 labels (as floats).
        val allOnes: Variable<T> = tf.variable(labels.asOutput().shape(), dtype)

        val labelsShifted = tf.math.sub(tf.math.mul(tf.constant(2f) as Operand<T>, labels), allOnes)


        return tf.withName(TRAINING_LOSS).math.mean(
            tf.math.mean(
                tf.nn.relu(
                    tf.math.sub(allOnes, tf.math.mul(labelsShifted, actual))
                ), tf.constant(0)
            )
            , tf.constant(0)
        )*/
    }
}

class HuberLoss<T : Number>(val delta: Float) : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        throw UnsupportedOperationException()

        /*val error = tf.math.sub(actual, labels)

    val deltaConst: Operand<T> = tf.dtypes.cast(tf.constant(delta), getDType()) // to actual.asOutput().dataType() in TF 2.x
    val point5: Operand<T> = tf.dtypes.cast(tf.constant(0.5), getDType())

    val abs_error: Operand<T> = tf.math.abs(error)
    val quadratic: Operand<T> = tf.math.minimum(abs_error, deltaConst)
    val linear: Operand<T> = tf.math.sub(abs_error, quadratic)

    val q2Point5: Operand<T> =
        tf.math.mul(point5, tf.math.mul(quadratic, quadratic))

    val deltaLinear: Operand<T> = tf.math.mul(deltaConst, linear)
    val loss: Operand<T> = tf.math.add(q2Point5, deltaLinear)

    val result: Operand<T> = Kmean(tf, loss, tf.constant(-1))
    return tf.withName(TRAINING_LOSS).identity(result)*/
    }
}

class LogLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, actual: Operand<T>, labels: Operand<T>, dtype: Class<T>): Operand<T> {
        throw UnsupportedOperationException()

        /* val epsilon = 1e-5f

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
         )*/
    }
}