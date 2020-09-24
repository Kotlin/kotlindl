package api.keras.metric

import api.keras.util.getDType
import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Metrics.
 */
enum class Metrics {
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
     * Computes the root of mean of squares of errors between labels and predictions.
     *
     * `metric = root(square(y_true - y_pred))`
     */
    RMSE;

    companion object {
        /** Converts enum value to sub-class of [Metric]. */
        fun convert(metricType: Metrics): Metric {
            return when (metricType) {
                ACCURACY -> Accuracy()
                MAE -> MAE()
                MSE -> MSE()
                RMSE -> RMSE()
            }
        }
    }
}

/**
 * @see [Metrics.ACCURACY]
 */
class Accuracy : Metric {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val predicted: Operand<Long> = tf.math.argMax(yPred, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(yTrue, tf.constant(1))

        return tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), getDType()), tf.constant(0))
    }
}

/**
 * @see [Metrics.MAE]
 */
class MAE : Metric {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val absoluteErrors = tf.math.abs(tf.math.sub(yPred, yTrue))
        return tf.reduceSum(tf.math.mean(absoluteErrors, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Metrics.MSE]
 */
class MSE : Metric {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val squaredError = tf.math.squaredDifference(yPred, yTrue)
        return tf.reduceSum(tf.math.mean(squaredError, tf.constant(-1)), tf.constant(0))
    }
}

/**
 * @see [Metrics.RMSE]
 */
class RMSE : Metric {
    override fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float> {
        val rootSquaredError = tf.math.sqrt(tf.math.squaredDifference(yPred, yTrue))
        return tf.reduceSum(tf.math.mean(rootSquaredError, tf.constant(-1)), tf.constant(0))
    }
}

