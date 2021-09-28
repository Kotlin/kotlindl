package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.image.colorOrder
import org.jetbrains.kotlinx.dl.dataset.image.imageType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.image.BufferedImage

/**
 * Converts provided image to the desired [ColorOrder].
 *
 * @property [colorOrder] target color order.
 */
public class Convert(public var colorOrder: ColorOrder = ColorOrder.BGR) : ImagePreprocessorBase() {
    override fun getOutputShape(inputShape: ImageShape?): ImageShape {
        return ImageShape(inputShape?.width, inputShape?.height, 3)
    }

    override fun apply(image: BufferedImage): BufferedImage {
        if (image.colorOrder() == colorOrder) return image
        val outputType = colorOrder.imageType()
        val result = BufferedImage(image.width, image.height, outputType)
        val graphics = result.createGraphics()
        graphics.drawImage(image, 0, 0, null)
        graphics.dispose()
        return result
    }
}