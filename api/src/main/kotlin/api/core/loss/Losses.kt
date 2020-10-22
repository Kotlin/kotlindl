/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package api.core.loss

import api.core.util.Kmean
import api.core.util.TRAINING_LOSS
import api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Loss functions.
 */
public enum class Losses {
    /**
     * Computes multi-dimensional sigmoid function (softmax) for outputs with logit operation.
     *
     * NOTE: This loss functions including another way of prediction output manipulations.
     */
    SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,

    /**
     * Computes the hinge loss between `y_true` and `y_pred`.
     *
     * `loss = maximum(1 - y_true * y_pred, 0)`
     *
     * `y_true` values are expected to be -1 or 1. If binary (0 or 1) labels are
     * provided we will convert them to -1 or 1.
     */
    HINGE_LOSS,

    /**
     * Computes the Huber loss between `y_true` and `y_pred`.
     *
     * For each value x in `error = y_true - y_pred`:
     *
     * ```
     * loss = 0.5 * x^2                  if |x| <= d
     * loss = 0.5 * d^2 + d * (|x| - d)  if |x| > d
     * ```
     */
    HUBER_LOSS,

    /**
     * Computes the Log loss between `y_true` and `y_pred`.
     */
    LOG_LOSS,

    /**
     * Computes the mean of absolute difference between labels and predictions.
     *
     * `loss = abs(y_true - y_pred)`
     */
    MAE,

    /**
     * Computes the mean of squares of errors between labels and predictions.
     *
     * `loss = square(y_true - y_pred)`
     */
    MSE,

    /**
     * Computes the root of mean of squares of errors between labels and predictions.
     *
     * `loss = root(square(y_true - y_pred))`
     */
    RMSE,

    /**
     * Computes the mean absolute percentage error between `y_true` and `y_pred`.
     *
     * `loss = 100 * abs(y_true - y_pred) / y_true`
     */
    MAPE,

    /**
     * Computes the mean squared logarithmic error between `y_true` and `y_pred`.
     *
     * `loss = square(log(y_true + 1.) - log(y_pred + 1.))`
     */
    MLSE,

    /**
     * Computes the Poisson loss between `y_true` and `y_pred`.
     *
     * `loss = y_pred - y_true * log(y_pred)`
     */
    POISSON;

    public companion object {
        /** Converts enum value to sub-class of [LossFunction]. */
        public fun convert(lossFunctionType: Losses): LossFunction {
            return when (lossFunctionType) {
                SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits()
                HINGE_LOSS -> HingeLoss()
                HUBER_LOSS -> HuberLoss(0.5f)
                LOG_LOSS -> LogLoss()
                MAE -> MAE()
                MSE -> MSE()
                RMSE -> RMSE()
                MAPE -> MAPE()
                MLSE -> MLSE()
                POISSON -> Poisson()
            }
        }
    }
}

/**
 * @see [Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS]
 */
public class SoftmaxCrossEntropyWithLogits : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(yPred, yTrue)

        return tf.withName(TRAINING_LOSS).math.mean(batchLoss.loss(), tf.constant(0))
    }
}

/**
 * @see [Losses.MAE]
 */
public class MAE : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val absoluteErrors = tf.math.abs(tf.math.sub(yPred, yTrue))
        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(absoluteErrors, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Losses.MSE]
 */
public class MSE : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val squaredError = tf.math.squaredDifference(yPred, yTrue)
        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(squaredError, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Losses.MAPE]
 */
public class MAPE : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val epsilon = 1e-5f

        val diff = tf.math.abs(
            tf.math.div(
                tf.math.sub(yTrue, yPred),
                tf.math.maximum(tf.math.abs(yTrue), tf.constant(epsilon))
            )
        )

        return tf.withName(TRAINING_LOSS).math.mul(
            tf.constant(100f),
            tf.reduceSum(tf.math.mean(diff, tf.constant(-1)), tf.constant(0))
        )
    }
}

/**
 * @see [Losses.MLSE]
 */
public class MLSE : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val epsilon = 1e-5f

        val firstLog = tf.math.log(tf.math.add(tf.math.maximum(yPred, tf.constant(epsilon)), tf.constant(1.0f)))
        val secondLog = tf.math.log(tf.math.add(tf.math.maximum(yTrue, tf.constant(epsilon)), tf.constant(1.0f)))

        return tf.withName(TRAINING_LOSS)
            .reduceSum(tf.math.mean(tf.math.squaredDifference(firstLog, secondLog), tf.constant(-1)), tf.constant(0))
    }
}


/**
 * @see [Losses.POISSON]
 */
public class Poisson : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val epsilon = 1e-5f
        val sub = tf.math.sub(yPred, tf.math.mul(yTrue, tf.math.log(tf.math.add(yPred, tf.constant(epsilon)))))

        return tf.withName(TRAINING_LOSS)
            .reduceSum(tf.math.mean(sub, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Losses.RMSE]
 */
public class RMSE : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val rootSquaredError = tf.math.sqrt(tf.math.squaredDifference(yPred, yTrue))
        return tf.withName(TRAINING_LOSS).reduceSum(tf.math.mean(rootSquaredError, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Losses.HINGE_LOSS]
 */
public class HingeLoss : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        // We first need to convert binary labels to -1/1 labels (as floats).
        val labelsShifted = tf.math.sub(tf.math.mul(tf.constant(2f) as Operand<Float>, yTrue), yTrue)

        return tf.withName(TRAINING_LOSS).reduceSum(
            tf.math.mean(
                tf.nn.relu(
                    tf.math.sub(yTrue, tf.math.mul(labelsShifted, yPred))
                ), tf.constant(-1)
            ), tf.constant(0)
        )
    }
}

/**
 * @see [Losses.HUBER_LOSS]
 *
 * @param [delta] Huber loss delta.
 */
// TODO: Huber is close to MAE/MSE via delta parameter: https://www.machinecurve.com/index.php/2019/10/12/using-huber-loss-in-keras/
public class HuberLoss(public val delta: Float) : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val error = tf.math.sub(yPred, yTrue)

        val deltaConst: Operand<Float> =
            tf.dtypes.cast(tf.constant(delta), getDType()) // to actual.asOutput().dataType() in TF 2.x
        val point5: Operand<Float> = tf.dtypes.cast(tf.constant(0.5), getDType())

        val absError: Operand<Float> = tf.math.abs(error)
        val quadratic: Operand<Float> = tf.math.minimum(absError, deltaConst)
        val linear: Operand<Float> = tf.math.sub(absError, quadratic)

        val q2Point5: Operand<Float> =
            tf.math.mul(point5, tf.math.mul(quadratic, quadratic))

        val deltaLinear: Operand<Float> = tf.math.mul(deltaConst, linear)
        val loss: Operand<Float> = tf.math.add(q2Point5, deltaLinear)

        val result: Operand<Float> = Kmean(tf, loss, tf.constant(-1))
        return tf.withName(TRAINING_LOSS).identity(tf.reduceSum(result, tf.constant(0)))
    }
}

/**
 * @see [Losses.LOG_LOSS]
 */
public class LogLoss : LossFunction {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val epsilon = 1e-5f

        val oneOp = tf.constant(1.0f) as Operand<Float>
        val minusOneOp = tf.constant(-1.0f) as Operand<Float>
        val epsilonOp = tf.constant(epsilon) as Operand<Float>

        val right = tf.math.mul(yTrue, tf.math.log(tf.math.add(yPred, epsilonOp)))
        val left =
            tf.math.mul(tf.math.log(tf.math.add(tf.math.sub(oneOp, yPred), epsilonOp)), tf.math.sub(oneOp, yTrue))

        val sum = tf.math.add(right, left)
        return tf.withName(TRAINING_LOSS).reduceSum(
            tf.math.mean(tf.math.mul(minusOneOp, sum), tf.constant(-1)),
            tf.constant(0)
        )
    }
}