package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import java.awt.image.BufferedImage

/**
 * This interface represents a custom transformation of a [FloatArray].
 *
 * It can be helpful for visualization of the result of augmentations applied to the image during training.
 *
 * Implementations of this interface can be provided to [ImageConverter.floatArrayToBufferedImage]
 * to transform input [FloatArray] before constructing [BufferedImage].
 * */
public fun interface ArrayTransform {
    /**
     * Invoke some transform for an [input] [FloatArray].
     * @param [input] [FloatArray] to transform.
     * */
    public fun invoke(input: FloatArray): FloatArray
}

/**
 * Each array element of an [input] is multiplied in-place by [scale] coefficient.
 * @param [input] [FloatArray] to multiply by [scale].
 * @param [scale] [Float] coefficient.
 */
public fun denormalizeInplace(
    input: FloatArray,
    scale: Float
): FloatArray {
    input.forEachIndexed { index, value -> input[index] = value * scale }
    return input
}
