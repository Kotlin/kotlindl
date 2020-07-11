package api.keras.layers

import api.TrainableTFModel
import api.keras.initializers.Initializer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Assign
import org.tensorflow.op.core.Variable

abstract class Layer<T : Number> {
    var name: String = ""

    lateinit var outputShape: LongArray

    lateinit var parentModel: TrainableTFModel<T>

    protected var dtype: Class<T> = getDType()

    /** Returns number of input parameters. */
    protected var fanIn: Int = 100

    /** Returns number of output parameters. */
    protected var fanOut: Int = 100

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
        initializers = initializers + Pair(name, initializer.apply(fanIn, fanOut, tf, variable, dtype, name))
        return variable
    }

    abstract fun getWeights(): List<Array<*>>

    abstract fun hasActivation(): Boolean

    abstract fun getParams(): Int
}