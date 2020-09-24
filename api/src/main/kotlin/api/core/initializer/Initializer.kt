package api.core.initializer

import api.core.shape.shapeOperand
import api.core.util.defaultAssignOpName
import api.core.util.defaultInitializerOpName
import org.tensorflow.Operand
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign

/**
 * Initializer base class: all initializers inherit this class.
 *
 * Initializers allow you to pre-specify an initialization strategy, encoded in
 * the Initializer object, without knowing the shape and dtype of the variable
 * being initialized.
 */
abstract class Initializer {
    /**
     * Adds an `Assign` Op to the graph to initialize
     * a tensorflow variable as specified by the initializer.
     *
     * @param [fanIn] The maximum number of inputs that an initializer can accept.
     * @param [fanOut] The maximum number of inputs that the output of an initializer can feed to other steps.
     * @param [tf] Tensorflow Ops Accessor
     * @param [input] Variable to initialize
     * @return Assign operand created.
     */
    fun apply(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        input: Operand<Float>,
        name: String
    ): Assign<Float> {
        return tf.withName(defaultAssignOpName(name)).assign(
            input, initialize(
                fanIn, fanOut, tf,
                shapeOperand(tf, input.asOutput().shape()), defaultInitializerOpName(name)
            )
        )
    }


    /**
     * Returns a Tensor object initialized as specified by the initializer.
     *
     * @param [fanIn] The maximum number of inputs that an initializer can accept.
     * @param [fanOut] The maximum number of inputs that the output of an initializer can feed to other steps.
     * @param [tf] Tensorflow Ops Accessor.
     * @param [shape] Shape of the tensor.
     * @param [name] Initializer name.
     */
    abstract fun initialize(
        fanIn: Int,
        fanOut: Int,
        tf: Ops,
        shape: Operand<Int>,
        name: String
    ): Operand<Float>
}