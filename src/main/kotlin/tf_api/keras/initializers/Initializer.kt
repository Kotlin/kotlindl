package tf_api.keras.initializers

import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import tf_api.keras.shapeOperand

abstract class Initializer {
    /**
     * Adds an `Assign` Op to the graph to initialize
     * a tensorflow variable as specified by the initializer.
     *
     * @param tf Tensorflow Ops Accessor
     * @param `in` Variable to initialize
     * @return Assign Operand created
     */
    fun <T : Number> apply(
        tf: Ops,
        input: Operand<T>,
        dtype: Class<T>
    ): Assign<T> {
        return tf.assign(input, initialize(tf, shapeOperand(tf, input.asOutput().shape()), dtype))
    }

    /**
     * Returns a Tensor object initialized as
     * specified by the initializer.
     *
     * @param tf    Tensorflow Ops Handle
     * @param shape Shape of the tensor
     */
    abstract fun <T : Number> initialize(
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>
    ): Operand<T>
}