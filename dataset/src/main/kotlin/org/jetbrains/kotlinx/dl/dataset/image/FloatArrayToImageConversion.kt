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

public object Conversions {
    public operator fun contains(pair: Pair<ArrayType, ColorMode>): Boolean {
        return pair in predefined
    }

    public operator fun get(pair: Pair<ArrayType, ColorMode>): ((FloatArray) -> FloatArray)? {
        return predefined[pair]
    }

    public val NormalizedGrayArrayToGrayImage: (FloatArray) -> FloatArray = {
        denormalizeInplace(it, scale = 255f)
    }

    public val NormalizedBGRArrayToBGRImage: (FloatArray) -> FloatArray = {
        denormalizeAndSwapChannels(it)
    }

    public val NormalizedBGRArrayToRGBImage: (FloatArray) -> FloatArray = {
        denormalizeAndSwapChannels(it)
    }

    public val NormalizedRGBArrayToBGRImage: (FloatArray) -> FloatArray = {
        denormalizeInplace(it, scale = 255f)
    }

    public val NormalizedRGBArrayToRGBImage: (FloatArray) -> FloatArray = {
        denormalizeInplace(it, scale = 255f)
    }

    public val RGBArrayToBGRImage: (FloatArray) -> FloatArray = {
        it
    }

    public val BGRArrayToRGBImage: (FloatArray) -> FloatArray = {
        ImageConverter.swapRandB(it)
        it
    }

    public val RGBArrayToRGBImage: (FloatArray) -> FloatArray = {
        it
    }

    public val BGRArrayToBGRImage: (FloatArray) -> FloatArray = {
        ImageConverter.swapRandB(it)
        it
    }

    private val predefined: Map<Pair<ArrayType, ColorMode>, (FloatArray) -> FloatArray> = mapOf(
        (ArrayType.NORMALIZED_GRAY to ColorMode.GRAYSCALE) to NormalizedGrayArrayToGrayImage,
        (ArrayType.NORMALIZED_BGR to ColorMode.BGR) to NormalizedBGRArrayToBGRImage,
        (ArrayType.NORMALIZED_BGR to ColorMode.RGB) to NormalizedBGRArrayToRGBImage,
        (ArrayType.BGR to ColorMode.BGR) to BGRArrayToBGRImage,
        (ArrayType.BGR to ColorMode.RGB) to BGRArrayToRGBImage,
        (ArrayType.NORMALIZED_RGB to ColorMode.RGB) to NormalizedRGBArrayToRGBImage,
        (ArrayType.NORMALIZED_RGB to ColorMode.BGR) to NormalizedRGBArrayToBGRImage,
        (ArrayType.RGB to ColorMode.BGR) to RGBArrayToBGRImage,
        (ArrayType.RGB to ColorMode.RGB) to RGBArrayToRGBImage
    )
}

public enum class ArrayType {
    NORMALIZED_GRAY,
    GRAY,
    NORMALIZED_RGB,
    RGB,
    NORMALIZED_BGR,
    BGR
}

public fun ArrayType.colorMode() : ColorMode {
    return when (this) {
        ArrayType.NORMALIZED_GRAY -> ColorMode.GRAYSCALE
        ArrayType.GRAY -> ColorMode.GRAYSCALE
        ArrayType.NORMALIZED_RGB -> ColorMode.RGB
        ArrayType.RGB -> ColorMode.RGB
        ArrayType.NORMALIZED_BGR -> ColorMode.BGR
        ArrayType.BGR -> ColorMode.BGR
    }
}

public fun ArrayType.isNormalized() : Boolean {
    return when (this) {
        ArrayType.NORMALIZED_GRAY -> true
        ArrayType.GRAY -> false
        ArrayType.NORMALIZED_RGB -> true
        ArrayType.RGB -> false
        ArrayType.NORMALIZED_BGR -> true
        ArrayType.BGR -> false
    }
}

public fun denormalizeInplace(
    input: FloatArray,
    scale: Float
): FloatArray {
    input.forEachIndexed { index, value -> input[index] = value * scale }
    return input
}

public fun denormalizeAndSwapChannels(input: FloatArray): FloatArray {
    denormalizeInplace(input, scale = 255f)
    ImageConverter.swapRandB(input)
    return input
}
