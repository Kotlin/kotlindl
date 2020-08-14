package api.keras.optimizers

import org.tensorflow.Operand
import org.tensorflow.op.Ops

class NoClipGradient<T : Number>() : ClipGradientAction<T>() {
    override fun clipGradient(tf: Ops, gradient: Operand<T>): Operand<T> {
        return gradient
    }
}

class ClipGradientByValue<T : Number>(private val clipValue: Float) : ClipGradientAction<T>() {
    override fun clipGradient(tf: Ops, gradient: Operand<T>): Operand<T> {
        return tf.clipByValue(gradient, tf.constant(-clipValue) as Operand<T>, tf.constant(clipValue) as Operand<T>)
    }

}

class ClipGradientByNorm<T : Number>(private val clipNormValue: Float) : ClipGradientAction<T>() {
    override fun clipGradient(tf: Ops, gradient: Operand<T>): Operand<T> {
        TODO("Is not implemented yet")
    }

}

abstract class ClipGradientAction<T : Number> {
    abstract fun clipGradient(tf: Ops, gradient: Operand<T>): Operand<T>
}
