package tf_api.blocks.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import tf_api.blocks.activations.Activations

class Conv2D<T : Number>(filterShape: IntArray, padding: IntArray, activation: Activations) : Layer<T>() {
    override fun addTFOperands(tf: Ops, inputShape: Shape) {
        TODO("Not yet implemented")
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        TODO("Not yet implemented")
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        TODO("Not yet implemented")
    }


}