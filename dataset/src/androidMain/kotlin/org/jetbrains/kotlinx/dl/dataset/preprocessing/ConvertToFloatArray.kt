package org.jetbrains.kotlinx.dl.dataset.preprocessing

import android.graphics.Bitmap
import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import java.nio.FloatBuffer

/**
 * Converts [Bitmap] to float array representation.
 */
// TODO support other image formats
// TODO address performance issues
public class ConvertToFloatArray : Operation<Bitmap, Pair<FloatArray, TensorShape>> {
    override fun apply(input: Bitmap): Pair<FloatArray, TensorShape> {
        val imgData = FloatBuffer.allocate(
            input.width * input.height * 3
        )
        imgData.rewind()
        val stride = input.width * input.height
        val bmpData = IntArray(stride.toInt())
        input.getPixels(bmpData, 0, input.width, 0, 0, input.width, input.height)
        var idx: Int = 0
        for (i in 0 until input.width) {
            for (j in 0 until input.height) {
                val pixelValue = bmpData[idx++]
                imgData.put((pixelValue shr 16 and 0xFF).toFloat())
                imgData.put((pixelValue shr 8 and 0xFF).toFloat())
                imgData.put((pixelValue and 0xFF).toFloat())
            }
        }

        imgData.rewind()
        return imgData.array() to TensorShape(input.width.toLong(), input.height.toLong(), 3)
    }

    override fun getFinalShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
