package org.jetbrains.kotlinx.dl.api.core.layer.reshaping

import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.tensorflow.Operand
import org.tensorflow.op.Ops

public abstract class AbstractZeroPadding(
    name:String
): Layer(name) {
    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingOperand = tf.constant(paddingArrayToTfFormat())
        val constantValue = tf.constant(0f)
        return tf.pad(input, paddingOperand, constantValue)
    }

    /**
     * This function helps in computing the padding operand i.e. normalizing the padding array
     * into a tensorflow format. This method will then be called in [forward] method that will be
     * further passed to tf.pad().
     */
    protected abstract fun paddingArrayToTfFormat(): Array<IntArray>
}
