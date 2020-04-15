package tf_api.blocks.layers

import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import org.tensorflow.op.core.Placeholder

class Source<T : Number>(vararg dims: Long) : Layer<T>() {
    private lateinit var input: Placeholder<T>
    private val packedDims: LongArray = dims

    override fun addTFOperands(tf: Ops, inputShape: Shape) {
        input = tf.placeholder(
            getDType(),
            Placeholder.shape(Shape.make(-1L, *packedDims))
        )
    }

    override fun computeOutputShape(inputShape: Shape): Shape {
        return input.asOutput().shape()
    }


    fun getDType(): Class<T> {
        return Float::class.javaObjectType as Class<T>
    }

    override fun transformInput(tf: Ops, input: Operand<T>): Operand<T> {
        return input
    }
}