package api.keras.layers

import api.core.KGraph
import api.keras.TrainableModel
import api.keras.initializers.Initializer
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Variable

/**
 * Base abstract class for all layers.
 *
 * @param [name] Layer name. Would be changed if empty during model compilation.
 */
abstract class Layer(var name: String) {
    /**
     * True, if layer's weights could be changed during training.
     * If false, layer's weights are frozen and could be changed during the training.
     */
    var isTrainable = true

    /** Output data tensor shape. */
    lateinit var outputShape: LongArray

    /** Model where this layer is used. */
    lateinit var parentModel: TrainableModel

    /** Basic DType for TensorFlow compatibility purposes. */
    protected var dtype: Class<Float> = getDType()

    /** Returns number of input parameters. */
    protected var fanIn: Int = Int.MIN_VALUE

    /** Returns number of output parameters. */
    protected var fanOut: Int = Int.MIN_VALUE

    abstract fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape)

    abstract fun computeOutputShape(inputShape: Shape): Shape

    abstract fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float>

    fun getDType(): Class<Float> {
        return Float::class.javaObjectType
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
        kGraph: KGraph,
        name: String,
        variable: Variable<Float>,
        initializer: Initializer
    ): Variable<Float> {
        require(fanIn != Int.MIN_VALUE) { "fanIn should be calculated before initialization for variable $name" }
        require(fanOut != Int.MIN_VALUE) { "fanOut should be calculated before initialization for variable $name" }

        val initOp = initializer.apply(fanIn, fanOut, tf, variable, name)
        kGraph.addLayerVariable(variable, isTrainable)
        kGraph.addInitializer(name, initOp)
        return variable
    }

    abstract fun getWeights(): List<Array<*>>

    abstract fun hasActivation(): Boolean

    abstract fun getParams(): Int
}