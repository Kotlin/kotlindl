package org.jetbrains.kotlinx.dl.dataset.image

import org.jetbrains.kotlinx.dl.dataset.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.Graphics2D
import java.awt.image.BufferedImage

internal fun BufferedImage.draw(block: (Graphics2D) -> Unit) {
    val graphics2D = createGraphics()
    try {
        block(graphics2D)
    } finally {
        graphics2D.dispose()
    }
}

internal fun BufferedImage.copy(): BufferedImage {
    val result = BufferedImage(width, height, type)
    copyData(result.raster)
    return result
}

internal fun BufferedImage.getShape(): ImageShape {
    return ImageShape(width.toLong(), height.toLong(), colorModel.numComponents.toLong())
}

internal fun BufferedImage.getTensorShape(): TensorShape {
    return TensorShape(width.toLong(), height.toLong(), colorModel.numComponents.toLong())
}
