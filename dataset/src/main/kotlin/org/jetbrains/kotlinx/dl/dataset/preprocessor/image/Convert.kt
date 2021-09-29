package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.colorMode
import org.jetbrains.kotlinx.dl.dataset.image.imageType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.image.BufferedImage

/**
 * Converts provided image to the desired [ColorMode].
 *
 * @property [colorMode] target color mode.
 */
public class Convert(public var colorMode: ColorMode = ColorMode.BGR) : ImagePreprocessorBase() {
    override fun getOutputShape(inputShape: ImageShape?): ImageShape {
        return ImageShape(inputShape?.width, inputShape?.height, 3)
    }

    override fun apply(image: BufferedImage): BufferedImage {
        if (image.colorMode() == colorMode) return image
        val outputType = colorMode.imageType()
        val result = BufferedImage(image.width, image.height, outputType)
        val graphics = result.createGraphics()
        graphics.drawImage(image, 0, 0, null)
        graphics.dispose()
        return result
    }
}