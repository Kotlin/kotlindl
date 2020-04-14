package tf_api.blocks.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable

abstract class Layer<T : Number> {
    var variables: Map<String, Variable<T>> = mapOf()

    var initializers: Map<String, Assign<T>> = mapOf()

    abstract fun addTFOperands(tf: Ops, inputShape: Shape)

    abstract fun computeOutputShape(inputShape: Shape): Shape

    abstract fun transformInput(tf: Ops, input: Operand<T>): Operand<T>
}