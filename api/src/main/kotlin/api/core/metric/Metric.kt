package api.core.metric

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Basic interface for all metric functions.
 */
public interface Metric {
    /**
     * Applies [Metric] to the [yPred] labels predicted by the model and known [yTrue] hidden during training.
     *
     * @param yPred The predicted values. shape = `[batch_size, d0, .. dN]`.
     * @param yTrue Ground truth values. Shape = `[batch_size, d0, .. dN]`.
     * @param [tf] TensorFlow graph API for building operations.
     */
    public fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float>
}