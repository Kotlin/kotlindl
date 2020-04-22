package tf_api.keras.metric

import org.tensorflow.Operand
import org.tensorflow.op.Ops

enum class Metrics {
    ACCURACY;

    companion object {
        fun <T : Number> convert(metricType: Metrics): Metric<T> {
            return when (metricType) {
                ACCURACY -> Accuracy()
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

