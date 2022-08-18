package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.getShape
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape.Companion.toTensorShape
import java.awt.image.BufferedImage

/**
 * Converts [BufferedImage] to float array representation.
 */
public class ConvertToFloatArray : Operation<BufferedImage, Pair<FloatArray, TensorShape>> {
    override fun apply(input: BufferedImage): Pair<FloatArray, TensorShape> {
        return ImageConverter.toRawFloatArray(input) to input.getShape().toTensorShape()
    }

    override fun getOutputShape(inputShape: TensorShape): TensorShape {
        return inputShape
    }
}
