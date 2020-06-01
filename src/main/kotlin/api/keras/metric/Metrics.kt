package api.keras.metric

import org.tensorflow.Operand
import org.tensorflow.op.Ops


// TODO: add more metrics from here https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/ops/metrics_impl.py#L862-L920
enum class Metrics {
    ACCURACY, MAE, MSE;

    companion object {
        fun <T : Number> convert(metricType: Metrics): Metric<T> {
            return when (metricType) {
                ACCURACY -> Accuracy()
                MAE -> MAE()
                MSE -> MSE()
            }
        }
    }
}

class Accuracy<T : Number>() : Metric<T> {
    override fun apply(tf: Ops, output: Operand<T>, label: Operand<T>, dtype: Class<T>): Operand<T> {
        val predicted: Operand<Long> = tf.math.argMax(output, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(label, tf.constant(1))

        return tf.math.mean(tf.dtypes.cast(tf.math.equal(predicted, expected), dtype), tf.constant(0))
    }
}

class MAE<T : Number>() : Metric<T> {
    override fun apply(tf: Ops, output: Operand<T>, label: Operand<T>, dtype: Class<T>): Operand<T> {
        val predicted: Operand<Long> = tf.math.argMax(output, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(label, tf.constant(1))

        val absoluteErrors = tf.math.abs(tf.math.sub(predicted, expected))

        return tf.math.mean(tf.dtypes.cast(absoluteErrors, dtype), tf.constant(0))
    }
}

class MSE<T : Number>() : Metric<T> {
    override fun apply(tf: Ops, output: Operand<T>, label: Operand<T>, dtype: Class<T>): Operand<T> {
        val predicted: Operand<Long> = tf.math.argMax(output, tf.constant(1))
        val expected: Operand<Long> = tf.math.argMax(label, tf.constant(1))

        val squaredError = tf.math.squaredDifference(predicted, expected)

        return tf.math.mean(tf.dtypes.cast(squaredError, dtype), tf.constant(0))
    }
}

