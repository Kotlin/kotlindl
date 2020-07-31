package api.keras.loss

import api.TRAINING_LOSS
import api.keras.Kmean
import api.keras.computeWeightedLoss
import api.keras.epsilon
import api.keras.squeezeOrExpandDimensions
import org.tensorflow.Operand
import org.tensorflow.op.Ops

enum class LossFunctions {
    SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
    ABSOLUTE_DIFFERENCE,
    POISSON,
    HINGE_LOSS,
    HUBER_LOSS,
    LOG_LOSS,
    MAE,
    MSE,
    MAPE;

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
                POISSON -> Poisson()
                MAPE -> MAPE()
            }
        }
    }
}

class Poisson<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val logPred = tf.math.log(tf.math.add(epsilon(tf), _yTrue));
        val resultingExpression = tf.math.sub(_yPred, tf.math.mul(_yTrue, logPred))

        val losses = Kmean(tf, resultingExpression, tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))
    }
}

class SoftmaxCrossEntropyWithLogits<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(yPred, yTrue)

        return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    }
}

class AbsoluteDifference<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val losses = tf.math.abs(tf.math.sub(yPred, yTrue))

        return tf.withName(TRAINING_LOSS).math.mean(tf.math.mean(losses, tf.constant(0)), tf.constant(0))
    }
}

class MAE<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val losses = Kmean(tf, tf.math.abs(tf.math.sub(_yPred, _yTrue)), tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))
    }
}

class MAPE<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val diff: Operand<T> = tf.math.abs(
            tf.math.div(
                tf.math.sub(yTrue, yPred),
                tf.math.maximum(tf.math.abs(yTrue), epsilon(tf))
            )
        )
        val losses = tf.math.mul(tf.constant(100f) as Operand<T>, Kmean(tf, diff, tf.constant(-1)))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))
    }
}

class MSE<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val losses = Kmean(tf, tf.math.squaredDifference(_yPred, _yTrue), tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))
    }
}

class HingeLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val sub = tf.math.sub(tf.constant(1f) as Operand<T>, tf.math.mul(_yPred, _yTrue))
        val maxResult = tf.math.maximum(tf.constant(0f) as Operand<T>, sub) as Operand<T>

        val losses = Kmean(tf, maxResult, tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))
    }
}

class HuberLoss<T : Number>(val delta: Float) : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

        val error = tf.math.sub(_yPred, _yTrue)

        val deltaConst: Operand<T> = tf.constant(delta) as Operand<T>
        val point5: Operand<T> = tf.constant(0.5f) as Operand<T>

        val abs_error: Operand<T> = tf.math.abs(error)
        val quadratic: Operand<T> = tf.math.minimum(abs_error, deltaConst)
        val linear: Operand<T> = tf.math.sub(abs_error, quadratic)

        val q2Point5: Operand<T> =
            tf.math.mul(point5, tf.math.mul(quadratic, quadratic))

        val deltaLinear: Operand<T> = tf.math.mul(deltaConst, linear)
        val loss: Operand<T> = tf.math.add(q2Point5, deltaLinear)

        val losses = Kmean(tf, loss, tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))

    }
}

class LogLoss<T : Number> : LossFunction<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        throw UnsupportedOperationException()
        /*val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions

         val epsilon = 1e-5f

         val oneOp = tf.constant(1.0f) as Operand<T>
         val minusOneOp = tf.constant(-1.0f) as Operand<T>
         val epsilonOp = tf.constant(epsilon) as Operand<T>

         val right = tf.math.mul(_yTrue, tf.math.log(tf.math.add(_yPred, epsilonOp)))
         val left = tf.math.mul(tf.math.log(tf.math.add(tf.math.sub(oneOp, _yPred), epsilonOp)), tf.math.sub(oneOp, _yTrue))
         val sum = tf.math.add(right, left)
        val loss = tf.math.mul(minusOneOp, sum)

        val losses = Kmean(tf, loss, tf.constant(0))
        return tf.withName(TRAINING_LOSS).identity(computeWeightedLoss(tf, losses))*/

    }
}