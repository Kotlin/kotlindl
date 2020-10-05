package api.core.layer

import api.core.KGraph
import api.core.shape.TensorShape
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Constant
import kotlin.math.abs

/**
 * Flattens an input tensor while preserving the batch axis (axis 0).
 *
 * @property [name] Custom layer name.
 * @constructor Creates [Flatten] object.
 */
public class Flatten(name: String = "") : Layer(name) {
    private lateinit var units: Constant<Int>

    override fun defineVariables(tf: Ops, kGraph: KGraph, inputShape: Shape) {
        val tensorShape = TensorShape(inputShape)
        val amountOfNeuronsInFlattenLayer = (tensorShape.numElements() / abs(tensorShape.size(0))).toInt()
        units = tf.constant(intArrayOf(-1, amountOfNeuronsInFlattenLayer))

        fanIn = tensorShape.numElements().toInt()
        fanOut = amountOfNeuronsInFlattenLayer
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        // leaves unknown dimensions unknown
        return Shape.make(TensorShape(inputShape).numElements())
    }

    override fun transformInput(tf: Ops, input: Operand<Float>): Operand<Float> {
        return tf.reshape(input, units)
    }

    override fun getWeights(): List<Array<*>> {
        return emptyList()
    }

    override fun hasActivation(): Boolean {
        return false
    }

    override fun getParams(): Int {
        return 0
    }

    override fun toString(): String {
        return "Flatten"
    }
}