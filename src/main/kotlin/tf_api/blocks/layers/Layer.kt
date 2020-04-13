package tf_api.blocks.layers

import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable

abstract class Layer<T : Number> {
    var variables: Map<String, Variable<T>> = mapOf()

    var initializers: Map<String, Assign<T>> = mapOf()

}