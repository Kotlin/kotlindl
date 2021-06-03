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
import org.tensorflow.op.core.Squeeze

/**
 * Max pooling operation for 1D temporal data (e.g. audio, timeseries).
 *
 * Downsamples the input by taking maximum value over a temporal window of size [poolSize].
 *
 * @property [poolSize] Size of the temporal pooling window.
 * @property [strides] The amount of shift for pooling window in each pooling step. If
 * `null`, it will default to [poolSize].
 * @property [padding] Padding strategy; can be either of [ConvPadding.VALID] which means no padding, or
 * [ConvPadding.SAME] which means padding the input equally such that the output has the same dimension
 * as the input.
 * @property [dataFormat] Data format of input; can be either of [CHANNELS_LAST], or [CHANNELS_FIRST].
 */
public class MaxPool1D(
    public val poolSize: Int = 2,
    public val strides: Int? = null,
    public val padding: ConvPadding = ConvPadding.VALID,
    public val dataFormat: String = CHANNELS_LAST,
    name: String = ""
) : Layer(name) {

    override val hasActivation: Boolean
        get() = false
    override val paramCount: Int
        get() = 0
    override val weights: Map<String, Array<*>>
        get() = emptyMap()

    init {
        require(dataFormat == CHANNELS_LAST || dataFormat == CHANNELS_FIRST) {
            "The dataFormat should be either \"$CHANNELS_LAST\" or \"$CHANNELS_FIRST\"."
        }

        require(padding == ConvPadding.VALID || padding == ConvPadding.SAME) {
            "The padding should be either ${ConvPadding.VALID} or ${ConvPadding.SAME}."
        }
    }

    override fun build(tf: Ops, kGraph: KGraph, inputShape: Shape) {}

    override fun computeOutputShape(inputShape: Shape): Shape {
        var steps = if (dataFormat == CHANNELS_LAST) inputShape.size(1) else inputShape.size(2)
        val strideValue = strides ?: poolSize
        steps = convOutputLength(steps, poolSize, padding, strideValue)
        return if (dataFormat == CHANNELS_LAST) {
            Shape.make(inputShape.size(0), steps, inputShape.size(2))
        } else {
            Shape.make(inputShape.size(0), inputShape.size(1), steps)
        }
    }

    override fun forward(
        tf: Ops,
        input: Operand<Float>,
        isTraining: Operand<Boolean>,
        numberOfLosses: Operand<Float>?
    ): Operand<Float> {
        val expandAxis = if (dataFormat == CHANNELS_LAST) 2 else 3
        val tfInput = tf.expandDims(input, tf.constant(expandAxis))
        val tfPoolSize = intArrayOf(1, 1, 1, 1)
        val tfStrides = intArrayOf(1, 1, 1, 1)
        /**
         * NOTE: we can use `MaxPool.Options` argument of `tf.nn.maxPool` to pass
         * the data format, as follows:
         * ```
         * val tfDataFormat = if (dataFormat == CHANNELS_LAST) "NHWC" else "NCHW_VECT_C"
         * tf.nn.maxPool(..., MaxPool.dataFormat(tfDataFormat))
         * ```
         * However, it seems it does not work for the case of "channels_first". So, instead
         * we are choosing to set the value of pool size and strides based on the data format.
         */
        tfPoolSize[expandAxis-1] = poolSize
        tfStrides[expandAxis-1] = strides ?: poolSize
        val tfPadding = padding.paddingName

        val maxPool = tf.nn.maxPool(
            tfInput,
            tf.constant(tfPoolSize),
            tf.constant(tfStrides),
            tfPadding
        )
        return tf.squeeze(maxPool, Squeeze.axis(listOf(expandAxis.toLong())))
    }

    override fun toString(): String =
        "MaxPool1D(poolSize=$poolSize, strides=$strides, padding=$padding, dataFormat=$dataFormat)"
}
