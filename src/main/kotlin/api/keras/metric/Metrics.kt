package api.keras.metric

import api.keras.Kmean
import api.keras.computeWeightedMetric
import api.keras.squeezeOrExpandDimensions
import org.tensorflow.Operand
import org.tensorflow.op.Ops


// TODO: https://github.com/tensorflow/tfjs/blob/master/tfjs-layers/src/metrics.ts
enum class Metrics {
    ACCURACY, MAE, MSE, MAPE, RMSE;

    companion object {
        fun <T : Number> convert(metricType: Metrics): Metric<T> {
            return when (metricType) {
                ACCURACY -> Accuracy()
                MAE -> MAE()
                MSE -> MSE()
                MAPE -> MAPE()
                RMSE -> RMSE()
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
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions
        val _losses = triada.losses

        val losses = Kmean(tf, tf.math.abs(tf.math.sub(_yPred, _yTrue)), tf.constant(-1))
        return computeWeightedMetric(tf, losses)
    }
}

class MSE<T : Number>() : Metric<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions
        val _losses = triada.losses

        val losses = Kmean(tf, tf.math.squaredDifference(_yPred, _yTrue), tf.constant(-1))
        return computeWeightedMetric(tf, losses)
    }
}

class RMSE<T : Number>() : Metric<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        val triada = squeezeOrExpandDimensions(tf, yTrue, yPred, null)
        val _yTrue = triada.labels
        val _yPred = triada.predictions
        val _losses = triada.losses

        val losses = Kmean(tf, tf.math.squaredDifference(_yPred, _yTrue), tf.constant(-1))
        return computeWeightedMetric(tf, losses)
    }
}

class MAPE<T : Number> : Metric<T> {
    override fun apply(tf: Ops, yPred: Operand<T>, yTrue: Operand<T>, dtype: Class<T>): Operand<T> {
        throw UnsupportedOperationException()
    }
}