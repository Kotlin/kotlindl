package api.keras.activations

import org.tensorflow.Operand
import org.tensorflow.op.Ops


interface Activation {
    /**
     * Applies the activation functions to the input [features] to produce the output.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [features] TensorFlow graph leaf node representing layer output before activation function.
     * @param [name] Activation name for TensorFlow graph building purposes.
     */
    fun apply(tf: Ops, features: Operand<Float>, name: String = ""): Operand<Float>
}