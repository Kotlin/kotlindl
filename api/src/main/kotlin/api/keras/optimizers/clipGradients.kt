package api.keras.optimizers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class NoClipGradient() : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        return gradient
    }
}

class ClipGradientByValue(private val clipValue: Float) : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        return tf.clipByValue(gradient, tf.constant(-clipValue) as Operand<Float>, tf.constant(clipValue))
    }

}

class ClipGradientByNorm(private val clipNormValue: Float) : ClipGradientAction() {
    override fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float> {
        TODO("Is not implemented yet")
    }

}

abstract class ClipGradientAction {
    abstract fun clipGradient(tf: Ops, gradient: Operand<Float>): Operand<Float>
}
