package api.keras.layers

import api.core.KGraph
import api.keras.TrainableModel
import api.keras.initializers.Initializer
import api.keras.util.getDType
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

    /**
     * Extend this function to define variables in layer.
     *
     * @param [tf] TensorFlow graph API for building operations.
     * @param [kGraph] [KGraph] to update it.
     * @param [inputShape] Input shape, result of [computeOutputShape] call from previous layer.
     */
    abstract fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape)

    /**
     * Computes output shape, based on [inputShape] and [Layer] type.
     */
    abstract fun computeOutputShape(inputShape: Shape): Shape

    /**
     * Builds main layer input transformation with [tf]. Depends on [Layer] type.
     */
    abstract fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float>

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

    /** Returns layer's weights. */
    abstract fun getWeights(): List<Array<*>>

    /** Returns True, if layer has internal activation function. */
    abstract fun hasActivation(): Boolean

    /** Returns amount of neurons. */
    abstract fun getParams(): Int
}