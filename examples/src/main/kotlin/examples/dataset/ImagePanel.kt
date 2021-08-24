package examples.dataset

import org.jetbrains.kotlinx.dl.api.extension.get3D
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import java.awt.Color
import java.awt.Graphics
import javax.swing.JPanel
import kotlin.math.max
import kotlin.math.min

class ImagePanel(private val image: FloatArray, private val imageShape: ImageShape) : JPanel() {
    override fun paint(graphics: Graphics) {
        for (i in 0 until imageShape.height!!.toInt()) { // rows
            for (j in 0 until imageShape.width!!.toInt()) { // columns
                val pixelWidth = 2
                val pixelHeight = 2
                val y = 100 + i * pixelWidth
                val x = 100 + j * pixelHeight

                val r = image.get3D(i, j, 2, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val g = image.get3D(i, j, 1, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val b = image.get3D(i, j, 0, imageShape.width!!.toInt(), imageShape.channels.toInt())
                val r1 = (min(1.0f, max(r * 0.8f, 0.0f)) * 255).toInt()
                val g1 = (min(1.0f, max(g * 0.8f, 0.0f)) * 255).toInt()
                val b1 = (min(1.0f, max(b * 0.8f, 0.0f)) * 255).toInt()
                val color = Color(r1, g1, b1)
                graphics.color = color
                graphics.fillRect(x, y, pixelWidth, pixelHeight)
                graphics.color = Color.BLACK
                graphics.drawRect(x, y, pixelWidth, pixelHeight)
            }
        }
    }
}