package tf_api.keras.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.keras.activations.Activations
import tf_api.keras.initializers.Initializers

class Dense<T : Number>(
    private val inputSize: Int,
    private val outputSize: Int,
    // activation function
    private val activation: Activations = Activations.Sigmoid,
    // initializers
    private val kernelInitializer: Initializers = Initializers.TRUNCATED_NORMAL,
    private val biasInitializer: Initializers = Initializers.ZEROS
) : Layer<T>() {
    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    override fun defineVariables(tf: Ops, inputShape: Shape) {
        // Compute shapes of kernel and bias matrices
        val kernelShape = Shape.make(inputShape.size(inputShape.numDimensions() - 1), inputSize.toLong())
        val biasShape = Shape.make(inputSize.toLong())


    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(outputSize.toLong());
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal: Operand<T> = tf.math.add(tf.linalg.matMul(input, kernel), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }

}