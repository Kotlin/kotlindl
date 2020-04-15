package tf_api.blocks.layers

import examples.PADDING_TYPE
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable
import tf_api.blocks.Initializer
import tf_api.blocks.activations.Activations

class Conv2D<T : Number>(private val filterShape: IntArray,
                         private val strides: LongArray,
                         private val activation: Activations = Activations.Relu,
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
        TODO("Not yet implemented")
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        val signal = tf.nn.biasAdd(tf.nn.conv2d(input, kernel, strides.toMutableList(), PADDING_TYPE), bias)
        return Activations.convert<T>(activation).apply(tf, signal)
    }


}