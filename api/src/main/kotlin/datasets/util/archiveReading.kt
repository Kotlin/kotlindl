package datasets.util

import api.keras.dataset.ImageDataset
import java.awt.image.BufferedImage
import java.awt.image.DataBufferByte
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.io.InputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipFile
import javax.imageio.ImageIO
import kotlin.experimental.and

private const val MILLS_IN_DAY = 86400000L

@Throws(IOException::class)
fun main() {
    val pathToLabels = "cifar10/images.zip"
    val realPathToLabels = ImageDataset::class.java.classLoader.getResource(pathToLabels)?.path.toString()
    val zipFile = ZipFile(realPathToLabels)
    val entries = zipFile.entries()

    val numOfPixels: Int = 32 * 32 * 3

    val images = Array(50000) { FloatArray(numOfPixels) }
    //val imageBuffer = ByteArray(numOfPixels)
    var cnt = 0

    while (entries.hasMoreElements()) {
        val entry = entries.nextElement() as ZipEntry
        val (imageByteArrays, image) = getImage(zipFile.getInputStream(entry))

        val pixels = (image.raster.dataBuffer as DataBufferByte).data

        images[cnt] =
            ImageDataset.toNormalizedVector(
                pixels
            )
        cnt++
    }

    println(images.size)
}

@Throws(IOException::class)
fun getImage(inputStream: InputStream, imageType: String = "png"): Pair<ByteArray, BufferedImage> {
    val image = ImageIO.read(inputStream)
    val baos = ByteArrayOutputStream()
    ImageIO.write(image, imageType, baos)
    return Pair(baos.toByteArray(), image)
}

class FastRGB internal constructor(image: BufferedImage) {
    var width: Int = image.width
    var height: Int = image.height
    private val hasAlphaChannel: Boolean = image.alphaRaster != null
    private var pixelLength: Int
    private val pixels: ByteArray = (image.raster.dataBuffer as DataBufferByte).data

    fun getRGB(x: Int, y: Int): ByteArray {
        var pos = y * pixelLength * width + x * pixelLength
        val rgb = ByteArray(4)
        if (hasAlphaChannel) rgb[3] = (pixels[pos++] and 0xFF.toByte()) // Alpha
        rgb[2] = (pixels[pos++] and 0xFF.toByte()) // Blue
        rgb[1] = (pixels[pos++] and 0xFF.toByte()) // Green
        rgb[0] = (pixels[pos++] and 0xFF.toByte()) // Red
        return rgb
    }

    init {
        pixelLength = 3
        if (hasAlphaChannel) pixelLength = 4
    }
}