/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.loss

import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.ReduceSum
import org.tensorflow.op.math.Mean

/**
 * Loss functions.
 */
public enum class Losses {
    /**
     * Computes multidimensional sigmoid function (softmax) for outputs with logit operation.
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
    HINGE,

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
    HUBER,

    /**
     * Computes the binary cross-entropy between `y_true` and `y_pred`.
     */
    BINARY_CROSSENTROPY,

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
    MSLE,

    /**
     * Computes the squared hinge loss between labels and predictions.
     *
     * `loss = square(maximum(1 - labels * predictions, 0))`
     */
    SQUARED_HINGE,

    /**
     * Computes the logarithm of the hyperbolic cosine of the prediction error.
     *
     * ```log(cosh(x))``` is approximately equal to ```(x ** 2) / 2``` for small ```x``` and
     * to ```abs(x) - log(2)``` for large ```x```. This means that logcosh works mostly
     * like the mean squared error, but will not be so strongly affected by the
     * occasional wildly incorrect prediction.
     */
    LOG_COSH,

    /**
     * Computes the Poisson loss between `y_true` and `y_pred`.
     *
     * `loss = y_pred - y_true * log(y_pred)`
     */
    POISSON;

    public companion object {
        /** Converts enum value to subclass of [LossFunction]. */
        public fun convert(lossFunctionType: Losses): LossFunction {
            return when (lossFunctionType) {
                SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS -> SoftmaxCrossEntropyWithLogits()
                HINGE -> Hinge()
                HUBER -> Huber()
                BINARY_CROSSENTROPY -> BinaryCrossEntropy()
                MAE -> MAE()
                MSE -> MSE()
                MAPE -> MAPE()
                MSLE -> MSLE()
                POISSON -> Poisson()
                SQUARED_HINGE -> SquaredHinge()
                LOG_COSH -> LogCosh()
            }
        }
    }
}

/**
 * @see [Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS]
 */
public class SoftmaxCrossEntropyWithLogits(reductionType: ReductionType = ReductionType.SUM) :
    LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val batchLoss = tf.nn.softmaxCrossEntropyWithLogits(yPred, yTrue)

        return tf.math.mean(batchLoss.loss(), tf.constant(0))
    }
}


/**
 * @see [Losses.MSE]
 */
public class MSE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val squaredError = tf.math.squaredDifference(yPred, yTrue)
        return meanOfLosses(tf, reductionType, squaredError, numberOfLosses)
    }
}

/**
 * @see [Losses.MAE]
 */
public class MAE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val absoluteErrors = tf.math.abs(tf.math.sub(yPred, yTrue))
        return meanOfLosses(tf, reductionType, absoluteErrors, numberOfLosses)
    }
}

/**
 * @see [Losses.MAPE]
 */
public class MAPE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-7f

        val diff = tf.math.abs(
            tf.math.div(
                tf.math.sub(yTrue, yPred),
                tf.math.maximum(tf.math.abs(yTrue), tf.constant(epsilon))
            )
        )

        return meanOfLosses(tf, reductionType, tf.math.mul(diff, tf.constant(100f)), numberOfLosses)
    }
}

/**
 * @see [Losses.MSLE]
 */
public class MSLE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-5f

        val firstLog = tf.math.log(tf.math.add(tf.math.maximum(yPred, tf.constant(epsilon)), tf.constant(1.0f)))
        val secondLog = tf.math.log(tf.math.add(tf.math.maximum(yTrue, tf.constant(epsilon)), tf.constant(1.0f)))

        val loss = tf.math.squaredDifference(firstLog, secondLog)

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}


/**
 * @see [Losses.POISSON]
 */
public class Poisson(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-5f
        val loss = tf.math.sub(yPred, tf.math.mul(yTrue, tf.math.log(tf.math.add(yPred, tf.constant(epsilon)))))

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}


/**
 * @see [Losses.HINGE]
 */
public class Hinge(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // We first need to convert binary labels to -1/1 labels (as floats).
        val two = tf.constant(2f)
        val one = tf.constant(1f)
        val zero = tf.constant(0f)

        val labelsShifted = tf.math.sub(tf.math.mul(two, yTrue), one)

        val loss: Operand<Float> =
            tf.math.maximum(tf.math.sub(one, tf.math.mul(labelsShifted, yPred)), zero)

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}

/**
 * @see [Losses.SQUARED_HINGE]
 */
public class SquaredHinge(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(
    reductionType
) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // We first need to convert binary labels to -1/1 labels (as floats).
        val two = tf.constant(2f)
        val one = tf.constant(1f)
        val zero = tf.constant(0f)

        val labelsShifted = tf.math.sub(tf.math.mul(two, yTrue), one)

        val loss: Operand<Float> = tf.math.square(
            tf.math.maximum(
                tf.math.sub(one, tf.math.mul(labelsShifted, yPred)),
                zero
            )
        )

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}

/**
 * @see [Losses.LOG_COSH]
 */
public class LogCosh(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(
    reductionType
) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val two = tf.constant(2f)
        val minusTwo = tf.constant(-2f)

        val diff = tf.math.sub(yPred, yTrue)
        val softplus = tf.math.softplus(tf.math.mul(minusTwo, diff))
        val loss = tf.math.sub(tf.math.add(diff, softplus), tf.math.log(two))

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}

/**
 * @see [Losses.HUBER]
 *
 * @param [delta] Huber loss delta.
 */
// TODO: Huber is close to MAE/MSE via delta parameter: https://www.machinecurve.com/index.php/2019/10/12/using-huber-loss-in-keras/
public class Huber(
    public val delta: Float = 1.0f,
    reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE
) :
    LossFunction(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
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

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}

/**
 * @see [Losses.BINARY_CROSSENTROPY]
 */
public class BinaryCrossEntropy(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : LossFunction(
    reductionType
) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-7f

        // Compute cross entropy from probabilities.
        val oneOp = tf.constant(1.0f) as Operand<Float>
        val minusOneOp = tf.constant(-1.0f) as Operand<Float>
        val epsilonOp = tf.constant(epsilon) as Operand<Float>
        val oneMinusEpsilonOp = tf.math.sub(oneOp, epsilonOp)

        // val clippedYPred = tf.clipByValue(yPred, epsilonOp, oneMinusEpsilonOp)
        // This section commented due to missed gradients for clipByValue op
        val clippedYPred = tf.math.minimum(
            oneMinusEpsilonOp,
            tf.math.maximum(epsilonOp, yPred)
        ) // probably takes 2 times memory for gradients instead of commented variant

        val right = tf.math.mul(yTrue, tf.math.log(tf.math.add(clippedYPred, epsilonOp)))
        val left =
            tf.math.mul(
                tf.math.log(tf.math.add(tf.math.sub(oneOp, clippedYPred), epsilonOp)),
                tf.math.sub(oneOp, yTrue)
            )

        val sum = tf.math.add(right, left)
        val loss = tf.math.mul(minusOneOp, sum)

        return meanOfLosses(tf, reductionType, loss, numberOfLosses)
    }
}

internal fun meanOfLosses(
    tf: Ops,
    reductionType: ReductionType,
    loss: Operand<Float>,
    numberOfLosses: Operand<Float>?
): Operand<Float> {
    val meanLoss = tf.math.mean(loss, tf.constant(-1), Mean.keepDims(false))

    // Eager session, correct calculation
    // numberOfLosses = tf.constant(TensorShape(loss.asOutput().shape()).numElements().toFloat())

    var totalLoss: Operand<Float> = tf.reduceSum(
        meanLoss,
        allAxes(tf, meanLoss),
        ReduceSum.keepDims(false)
    )

    if (reductionType == ReductionType.SUM_OVER_BATCH_SIZE) {
        check(numberOfLosses != null) { "Operand numberOfLosses must be not null." }

        totalLoss = safeMean(
            tf,
            loss,
            numberOfLosses
        )
    }

    return totalLoss
}

internal fun safeMean(tf: Ops, loss: Operand<Float>, numElements: Operand<Float>): Operand<Float> {
    val totalLoss: Operand<Float> = tf.reduceSum(loss, allAxes(tf, loss))
    return tf.math.divNoNan(totalLoss, numElements)
}

internal fun allAxes(tf: Ops, op: Operand<Float>): Operand<Int> {
    val rank = op.asOutput().shape().numDimensions()
    return if (rank != -1) {
        val axes = IntArray(rank)
        for (i in 0 until rank) {
            axes[i] = i
        }
        tf.constant(axes)
    } else {
        tf.range(tf.constant(0), tf.rank(op), tf.constant(1))
    }
}
