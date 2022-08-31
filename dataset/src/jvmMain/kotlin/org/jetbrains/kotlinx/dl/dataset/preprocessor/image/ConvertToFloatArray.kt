package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.getTensorShape
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import java.awt.image.BufferedImage

/**
 * Converts [BufferedImage] to float array representation.
 */
public class ConvertToFloatArray : Operation<BufferedImage, Pair<FloatArray, TensorShape>> {
    override fun apply(input: BufferedImage): Pair<FloatArray, TensorShape> {
        return ImageConverter.toRawFloatArray(input) to input.getTensorShape()
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
