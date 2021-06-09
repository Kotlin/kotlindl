package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.core.shape.shapeOperand
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import java.util.*

/**
 * Max pooling operation for 3D data (spatial or spatio-temporal).
 * NOTE: Works with tensors which must have rank 5 (batch, depth, height, width, channels).
 * @property [poolSize] The size of the sliding window for each dimension of input tensor (pool batch, pool depth ,pool height, pool width, pool channels).
 * Usually, pool batch and pool channels are equal to 1.
 * @property [strides] Strides of the pooling operation for each dimension of input tensor.
 * @property [padding] The padding method, either 'valid' or 'same'.
 * @property [name] Custom layer name.
 * @constructor Creates [MaxPool2D] object.
 */
public class MaxPool3D(
    public var poolSize: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public var strides: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name) {

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        // TODO add dataFormat support
        var len_dim1: Long = inputShape.size(1)
        var len_dim2: Long = inputShape.size(2)
        var len_dim3: Long = inputShape.size(3)

        len_dim1 = convOutputLength(len_dim1, poolSize[1], padding, strides[1])
        len_dim2 = convOutputLength(len_dim2, poolSize[2], padding, strides[2])
        len_dim3 = convOutputLength(len_dim3, poolSize[3], padding, strides[3])

        return Shape.make(inputShape.size(0), len_dim1, len_dim2, len_dim3, inputShape.size(4))
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        // TODO add dataFormat support
        val paddingName = padding.paddingName
        var tfPoolSize = Arrays.stream(poolSize).asLongStream().toArray();
        var tfStrides = Arrays.stream(strides).asLongStream().toArray();
        var tfInput: Operand<Float> = input
        var output = tf.nn.maxPool3d(tfInput, tfPoolSize.toList(), tfStrides.toList(), paddingName)
        return output
    }

    override var weights: Map<String, Array<*>>
        get() = emptyMap()
        set(value) = assignWeights(value)

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "MaxPool3D(poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding)"
    }

}
