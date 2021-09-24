package examples.dataset

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.Color
import java.awt.Dimension
import java.awt.Graphics
import java.awt.image.BufferedImage
import javax.swing.JPanel

class ImagePanel(image: FloatArray, imageShape: ImageShape) : JPanel() {
    private val bufferedImage = image.toBufferedImage(imageShape)

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

private fun FloatArray.toBufferedImage(imageShape: ImageShape): BufferedImage {
    val result = BufferedImage(imageShape.width!!.toInt(), imageShape.height!!.toInt(), BufferedImage.TYPE_INT_RGB)
    for (i in 0 until imageShape.height!!.toInt()) { // rows
        for (j in 0 until imageShape.width!!.toInt()) { // columns
            val r = get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            val g = get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            val b = get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels.toInt()).coerceIn(0f, 1f)
            result.setRGB(j, i, Color(r, g, b).rgb)
        }
    }
    return result
}