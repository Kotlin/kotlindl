/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.metric

import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.loss.ReductionType
import org.jetbrains.kotlinx.dl.api.core.loss.allAxes
import org.jetbrains.kotlinx.dl.api.core.loss.safeMean
import org.jetbrains.kotlinx.dl.api.core.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.ReduceSum
import org.tensorflow.op.math.Mean

/**
 * Metrics.
 */
public enum class Metrics {
    /**
     * Computes the rate of true answers.
     *
     * `metric = sum(y_true == y_pred)`
     */
    ACCURACY,

    /**
     * Computes the mean of absolute difference between labels and predictions.
     *
     * `metric = abs(y_true - y_pred)`
     */
    MAE,

    /**
     * Computes the mean of squares of errors between labels and predictions.
     *
     * `metric = square(y_true - y_pred)`
     */
    MSE,

    /**
     * Computes the mean squared logarithmic error between `y_true` and `y_pred`.
     *
     * `loss = square(log(y_true + 1.) - log(y_pred + 1.))`
     */
    MSLE;

    public companion object {
        /** Converts enum value to sub-class of [Metric]. */
        public fun convert(metricType: Metrics): Metric {
            return when (metricType) {
                ACCURACY -> Accuracy()
                MAE -> MAE()
                MSE -> MSE()
                MSLE -> MSLE()
            }
        }

        /** Converts sub-class of [Metric] to enum value. */
        public fun convertBack(metric: Metric): Metrics {
            return when (metric) {
                is Accuracy -> ACCURACY
                is org.jetbrains.kotlinx.dl.api.core.metric.MAE -> MAE
                is org.jetbrains.kotlinx.dl.api.core.metric.MSE -> MSE
                is org.jetbrains.kotlinx.dl.api.core.metric.MSLE -> MSLE
                else -> ACCURACY
            }
        }
    }
}

/**
 * @see [Metrics.ACCURACY]
 */
public class Accuracy : Metric(reductionType = ReductionType.SUM_OVER_BATCH_SIZE) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        val predicted: Operand<Long> = tf.math.argMax(yPred, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(yTrue, tf.constant(1))

        return tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), getDType()), tf.constant(0))
    }
}

/**
 * @see [Losses.MSE]
 */
public class MSE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : Metric(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        val squaredError = tf.math.squaredDifference(yPred, yTrue)
        return meanOfMetrics(tf, reductionType, squaredError, numberOfLabels, "Metric_MSE")
    }
}

/**
 * @see [Losses.MAE]
 */
public class MAE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : Metric(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        val absoluteErrors = tf.math.abs(tf.math.sub(yPred, yTrue))
        return meanOfMetrics(tf, reductionType, absoluteErrors, numberOfLabels, "Metric_MAE")
    }
}

/**
 * @see [Losses.MAPE]
 */
public class MAPE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : Metric(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-7f

        val diff = tf.math.abs(
            tf.math.div(
                tf.math.sub(yTrue, yPred),
                tf.math.maximum(tf.math.abs(yTrue), tf.constant(epsilon))
            )
        )

        return meanOfMetrics(tf, reductionType, tf.math.mul(diff, tf.constant(100f)), numberOfLabels, "Metric_MAPE")
    }
}

/**
 * @see [Losses.MSLE]
 */
public class MSLE(reductionType: ReductionType = ReductionType.SUM_OVER_BATCH_SIZE) : Metric(reductionType) {
    override fun apply(
        tf: Ops,
        yPred: Operand<Float>,
        yTrue: Operand<Float>,
        numberOfLabels: Operand<Float>?
    ): Operand<Float> {
        val epsilon = 1e-5f

        val firstLog = tf.math.log(tf.math.add(tf.math.maximum(yPred, tf.constant(epsilon)), tf.constant(1.0f)))
        val secondLog = tf.math.log(tf.math.add(tf.math.maximum(yTrue, tf.constant(epsilon)), tf.constant(1.0f)))

        val loss = tf.math.squaredDifference(firstLog, secondLog)

        return meanOfMetrics(tf, reductionType, loss, numberOfLabels, "Metric_MSLE")
    }
}

internal fun meanOfMetrics(
    tf: Ops,
    reductionType: ReductionType,
    metric: Operand<Float>,
    numberOfLabels: Operand<Float>?,
    metricName: String
): Operand<Float> {
    val meanMetric = tf.math.mean(metric, tf.constant(-1), Mean.keepDims(false))

    var totalMetric: Operand<Float> = tf.reduceSum(
        meanMetric,
        allAxes(tf, meanMetric),
        ReduceSum.keepDims(false)
    )

    if (reductionType == ReductionType.SUM_OVER_BATCH_SIZE) {
        check(numberOfLabels != null) { "Operand numberOfLosses must be not null." }

        totalMetric = safeMean(
            tf,
            metric,
            numberOfLabels
        )
    }

    return tf.withName(metricName).identity(totalMetric)
}
