package examples.io

import examples.transferlearning.getFileFromResource
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertDoesNotThrow
import javax.imageio.ImageIO
import javax.imageio.ImageReader


class IOTestSuite {
    private val unsupportedJpegsFolder = getFileFromResource("jpeg_unsupported")

    @Test
    fun testJvmUnsupportedJpegLoading() {
        val imageFiles = unsupportedJpegsFolder.walk().filter { it.isFile }

        for (imageFile in imageFiles) {
            assertDoesNotThrow("Failed to load image file $imageFile") {
                imageFile.inputStream().use { inputStream -> ImageConverter.toBufferedImage(inputStream) }
            }
        }
    }

    @Test
    fun testImageioJpegPluginIsUsed() {
        val readers: Iterator<ImageReader> = ImageIO.getImageReadersByFormatName("JPEG")
        val defaultReader = readers.next()
        assert("com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageReader" in defaultReader.toString()) { "Default jpeg reader should be com.twelvemonkeys.imageio.plugins.jpeg.JPEGImageReader" }
    }
}
