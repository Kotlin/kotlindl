package tf_api.keras.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable
import tf_api.keras.initializers.Initializer

abstract class Layer<T : Number> {
    protected var dtype: Class<T> = getDType()

    var variables: Map<String, Variable<T>> = mutableMapOf()

    var initializers: Map<String, Assign<T>> = mutableMapOf()

    abstract fun defineVariables(tf: Ops, inputShape: Shape)

    abstract fun computeOutputShape(inputShape: Shape): Shape

    abstract fun transformInput(tf: Ops, input: Operand<T>): Operand<T>

    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }

    /**
     * Adds a new weight tensor to the layer
     *
     * @param name     variable name
     * @param variable variable to add
     * @return the created variable.
     */
    protected fun addWeight(
        tf: Ops,
        name: String,
        variable: Variable<T>,
        initializerName: String,
        initializer: Initializer<T>
    ): Variable<T> {
        variables = variables + Pair(name, variable)
        initializers = initializers + Pair(initializerName, initializer.apply(tf, variable, dtype))
        return variable
    }
}