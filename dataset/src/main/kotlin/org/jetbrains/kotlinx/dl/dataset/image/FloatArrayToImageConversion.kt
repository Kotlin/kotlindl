package org.jetbrains.kotlinx.dl.dataset.image

/**
 * This interface represents preprocessing logic for [FloatArray] image.
 * NOTE: by convention output array should be in RGB order.
 *       Thus, it can be used with floatArrayToBufferedImage
 * @see org.jetbrains.kotlinx.dl.dataset.image.ImageConverter.floatArrayToBufferedImage
 * */
public fun interface Conversion {
    public fun invoke(input: FloatArray): FloatArray
}

public fun denormalizeInplace(
    input: FloatArray,
    scale: Float
): FloatArray {
    input.forEachIndexed { index, value -> input[index] = value * scale }
    return input
}
