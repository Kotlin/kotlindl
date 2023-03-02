package examples.dataset

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.dl.api.preprocessing.pipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ColorMode
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter.floatArrayToBufferedImage
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter.toRawFloatArray
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.convert
import org.jetbrains.kotlinx.dl.visualization.swing.ImagePanel
import org.jetbrains.kotlinx.dl.visualization.swing.showFrame
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ndarray
import org.jetbrains.kotlinx.multik.ndarray.data.D3Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.toFloatArray
import java.awt.image.BufferedImage
import java.io.File


/**
 * This example demonstrates how to implement custom preprocessing operations and use them in the pipeline.
 */

private class ConvertToNdArray : Operation<BufferedImage, D3Array<Float>> {
    override fun apply(input: BufferedImage): D3Array<Float> {
        val tensorShape = intArrayOf(input.height, input.width, 3)
        val data = toRawFloatArray(input)

        return mk.ndarray(data.toList(), tensorShape)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}

private fun <I> Operation<I, BufferedImage>.toNdArray(block: ConvertToNdArray.() -> Unit): Operation<I, D3Array<Float>> {
    return PreprocessingPipeline(this, ConvertToNdArray().apply(block))
}

private class SwapChannels : Operation<D3Array<Float>, D3Array<Float>> {
    override fun apply(input: D3Array<Float>): D3Array<Float> {
        val (w, h, _) = input.shape
        for (i in 0 until w) {
            for (j in 0 until h) {
                val tmp = input[i, j, 0]
                input[i, j, 0] = input[i, j, 2]
                input[i, j, 2] = tmp
            }
        }
        return input
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        val dims = inputShape.dims()
        return TensorShape(dims[0], dims[1], dims[2])
    }
}

private fun <I> Operation<I, D3Array<Float>>.swapChannels(block: SwapChannels.() -> Unit): Operation<I, D3Array<Float>> {
    return PreprocessingPipeline(this, SwapChannels().apply(block))
}

private class Rotate90Ccw : Operation<D3Array<Float>, D3Array<Float>> {
    override fun apply(input: D3Array<Float>): D3Array<Float> {
        return input.transpose(1, 0, 2)
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        val dims = inputShape.dims()
        return TensorShape(dims[1], dims[0], dims[2])
    }
}


private fun <I> Operation<I, D3Array<Float>>.rotate90Ccw(block: Rotate90Ccw.() -> Unit): Operation<I, D3Array<Float>> {
    return PreprocessingPipeline(this, Rotate90Ccw().apply(block))
}

fun main() {
    val imageResource = Operation::class.java.getResource("/datasets/detection/image3.jpg")
    val file = File(imageResource!!.toURI())

    val image = ImageConverter.toBufferedImage(file)

    val pipeline = pipeline<BufferedImage>()
        .convert { colorMode = ColorMode.BGR }
        .toNdArray {}
        .swapChannels {}
        .rotate90Ccw {}

    val ndArray = pipeline.apply(image)
    val (height, width, _) = ndArray.shape

    val result = floatArrayToBufferedImage(
        ndArray.toFloatArray(),
        width, height, ColorMode.RGB,
        isNormalized = false
    )

    showFrame("Result of custom transformation", ImagePanel(result))
}
