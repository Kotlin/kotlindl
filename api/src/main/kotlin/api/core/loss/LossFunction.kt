package api.core.loss

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * Basic interface for all loss functions.
 */
public interface LossFunction {
    /**
     * Applies [LossFunction] to the [yPred] labels predicted by the model and known [yTrue] hidden during training.
     *
     * @param yPred The predicted values. shape = `[batch_size, d0, .. dN]`.
     * @param yTrue Ground truth values. Shape = `[batch_size, d0, .. dN]`, except
     * sparse loss functions such as sparse categorical crossentropy where
     * shape = `[batch_size, d0, .. dN-1]`.
     * @param [tf] TensorFlow graph API for building operations.
     */
    public fun apply(tf: Ops, yPred: Operand<Float>, yTrue: Operand<Float>): Operand<Float>
}
