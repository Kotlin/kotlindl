package examples.dataset

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.imageType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import javax.swing.JPanel

class ImagePanel(image: FloatArray, imageShape: ImageShape, colorMode: ColorMode) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape, colorMode)

    override fun paint(graphics: Graphics) {
        super.paint(graphics)
        val x = (size.width - bufferedImage.width) / 2
        val y = (size.height - bufferedImage.height) / 2
        graphics.drawImage(bufferedImage, x, y, null)
    }

    override fun getPreferredSize(): Dimension {
        return Dimension(bufferedImage.width, bufferedImage.height)
    }
}

private fun FloatArray.toBufferedImage(imageShape: ImageShape, colorMode: ColorMode): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), colorMode.imageType())
    val rgbArray = copyOf().also {
        if (colorMode == ColorMode.BGR) ImageConverter.swapRandB(it)
    }
    rgbArray.forEachIndexed { index, value -> rgbArray[index] = value * 255f }
    result.raster.setPixels(0, 0, result.width, result.height, rgbArray)
    return result
}