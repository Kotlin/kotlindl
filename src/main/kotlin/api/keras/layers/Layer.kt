package api.keras.layers

import api.KGraph
import api.TrainableTFModel
import api.keras.initializers.Initializer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

abstract class Layer<T : Number> {
    var name: String = ""

    var isTrainable = true

    lateinit var outputShape: LongArray

    lateinit var parentModel: TrainableTFModel<T>

    protected var dtype: Class<T> = getDType()

    /** Returns number of input parameters. */
    protected var fanIn: Int = Int.MIN_VALUE

    /** Returns number of output parameters. */
    protected var fanOut: Int = Int.MIN_VALUE

    abstract fun defineVariables(tf: Ops, kGraph: KGraph<T>, inputShape: Shape)

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
        kGraph: KGraph<T>,
        name: String,
        variable: Variable<T>,
        initializerName: String,
        initializer: Initializer<T>
    ): Variable<T> {
        require(fanIn != Int.MIN_VALUE) { "fanIn should be calculated before initialization for variable $name" }
        require(fanOut != Int.MIN_VALUE) { "fanOut should be calculated before initialization for variable $name" }

        val initOp = initializer.apply(fanIn, fanOut, tf, variable, dtype, name)
        kGraph.addVariable(variable, isTrainable)
        kGraph.addInitializer(name, initOp)
        return variable
    }

    abstract fun getWeights(): List<Array<*>>

    abstract fun hasActivation(): Boolean

    abstract fun getParams(): Int
}