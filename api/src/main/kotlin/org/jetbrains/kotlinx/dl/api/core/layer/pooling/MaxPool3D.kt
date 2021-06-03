package org.jetbrains.kotlinx.dl.api.core.layer.pooling

import org.jetbrains.kotlinx.dl.api.core.KGraph
import org.jetbrains.kotlinx.dl.api.core.layer.Layer
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.shape.convOutputLength
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_FIRST
import org.jetbrains.kotlinx.dl.api.inference.keras.CHANNELS_LAST
import org.tensorflow.Operand
import org.tensorflow.Shape
import org.tensorflow.op.Ops
import java.util.*

class MaxPool3D(
    public val dataFormat: String = CHANNELS_LAST,
    public var poolSize: IntArray = intArrayOf(1, 2, 2, 2, 1),
    public var strides: IntArray? = null,
    public val padding: ConvPadding = ConvPadding.VALID,
    name: String = ""
) : Layer(name){

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var len_dim1:Long
        var len_dim2:Long
        var len_dim3:Long
        if (dataFormat== CHANNELS_FIRST){
            len_dim1 = inputShape.size(2)
            len_dim2 = inputShape.size(3)
            len_dim3 = inputShape.size(4)
        }else{
            len_dim1 = inputShape.size(1)
            len_dim2 = inputShape.size(2)
            len_dim3 = inputShape.size(3)
        }
        len_dim1 = convOutputLength(len_dim1, poolSize[1], padding, strides?.get(1) ?: poolSize[1] )
        len_dim2 = convOutputLength(len_dim2, poolSize[2], padding, strides?.get(2) ?: poolSize[2])
        len_dim3 = convOutputLength(len_dim3, poolSize[3], padding, strides?.get(3) ?: poolSize[3])
        if(dataFormat == CHANNELS_FIRST){
            return Shape.make(inputShape.size(0), inputShape.size(1), len_dim1, len_dim2, len_dim3)
        }else{
            return Shape.make(inputShape.size(0), len_dim1,len_dim2,len_dim3,inputShape.size(4))
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val paddingName = padding.paddingName
        var tfPoolSize = Arrays.stream(poolSize).asLongStream().toArray();
        var tfStrides = Arrays.stream(strides ?: poolSize).asLongStream().toArray();
//        if(dataFormat== CHANNELS_FIRST) input = transpose()
        return tf.nn.maxPool3d(input,tfPoolSize.toList(), tfStrides.toList(), paddingName,  )
    }

    override val weights: Map<String, Array<*>> get() = emptyMap()

    override val hasActivation: Boolean get() = false

    override val paramCount: Int get() = 0

    override fun toString(): String {
        return "MaxPool3D(dataFormat=$dataFormat,poolSize=${poolSize.contentToString()}, strides=${strides.contentToString()}, padding=$padding)"
    }

}
