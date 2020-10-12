package api.core.optimizer

import org.tensorflow.Operand
import org.tensorflow.op.Ops

/**
 * No gradient clipping. Gradients go forward without any cnahges.
 */
public class NoClipGradient : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        return gradient
    }
}

/**
 * Clips gradient value to [clipValue] if gradient value more than [clipValue] and to -[clipValue]
 * if gradient value less than -[clipValue].
 *
 * @property [clipValue] Value limit for gradient.
 */
public class ClipGradientByValue(private val clipValue: Float) : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        return tf.clipByValue(gradient, tf.constant(-clipValue) as Operand<Float>, tf.constant(clipValue))
    }
}

/**
 * Clips gradient with pre-defined [clipNormValue] norm.
 *
 * NOTE: Is not supported yet.
 */
public class ClipGradientByNorm(private val clipNormValue: Float) : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        throw UnsupportedOperationException("Is not implemented yet!")
    }
}

/**
 * Base abstract class for approaches to clip gradient values from step to step in optimizer.
 */
public abstract class ClipGradientAction {
    /**
     * Clips [gradient].
     *
     * @param [tf] TensorFlow graph API for building operations.
     */
    public abstract fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float>
}
