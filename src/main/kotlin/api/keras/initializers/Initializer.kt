package api.keras.initializers

import api.keras.shape.shapeOperand
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign

abstract class Initializer<T : Number> {
    /**
     * Adds an `Assign` Op to the graph to initialize
     * a tensorflow variable as specified by the initializer.
     *
     * @param tf Tensorflow Ops Accessor
     * @param `in` Variable to initialize
     * @return Assign Operand created
     */
    fun apply(
        funIn: Int,
        funOut: Int,
        tf: Ops,
        input: Operand<T>,
        dtype: Class<T>,
        name: String
    ): Assign<T> {
        return tf.withName("Assign_$name").assign(
            input, initialize(
                funIn, funOut, tf,
                shapeOperand(tf, input.asOutput().shape()), dtype, "Init_$name"
            )
        )
    }

    /**
     * Returns a Tensor object initialized as
     * specified by the initializer.
     *
     * @param tf    Tensorflow Ops Handle
     * @param shape Shape of the tensor
     */
    abstract fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        dtype: Class<T>,
        name: String
    ): Operand<T>
}