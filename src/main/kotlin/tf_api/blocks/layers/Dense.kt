package tf_api.blocks.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.blocks.Initializer
import tf_api.blocks.activations.Activations

class Dense<T : Number>(
    private val inputSize: Int,
    private val outputSize: Int,
    // activation function
    private val activation: Activations = Activations.Sigmoid,
    // initializers
    private val kernelInitializer: Initializer = Initializer.TRUNCATED_NORMAL,
    private val biasInitializer: Initializer = Initializer.ZEROS
) : Layer<T>() {
    // weight tensors
    private lateinit var kernel: Variable<T>
    private lateinit var bias: Variable<T>

    override fun addTFOperands(tf: Ops, inputShape: Shape) {
        TODO("Not yet implemented")
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return Shape.make(outputSize.toLong());
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal: Operand<T> = tf.math.add(tf.linalg.matMul(input, kernel), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }

}