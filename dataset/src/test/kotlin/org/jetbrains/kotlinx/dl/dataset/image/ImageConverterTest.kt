package org.jetbrains.kotlinx.dl.dataset.image

import org.jetbrains.kotlinx.dl.dataset.preprocessor.PreprocessingImageTest.Companion.setRGB
import org.jetbrains.kotlinx.dl.dataset.preprocessor.getShape
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage

class ImageConverterTest {
    @Test
    fun bgr2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_3BYTE_BGR, ColorOrder.BGR)

    @Test
    fun bgr2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_3BYTE_BGR, ColorOrder.RGB)

    @Test
    fun ibgr2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_BGR, ColorOrder.BGR)

    @Test
    fun ibgr2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_BGR, ColorOrder.RGB)

    @Test
    fun rgb2bgrImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_RGB, ColorOrder.BGR)

    @Test
    fun rgb2rgbImageToFloatsTest() = imageToFloatsTest(BufferedImage.TYPE_INT_RGB, ColorOrder.RGB)

    private fun imageToFloatsTest(sourceImageType: Int, targetColorOrder: ColorOrder) {
        val sourceImage = BufferedImage(2, 2, sourceImageType)
        val color1 = Color(50, 150, 200)
        val color2 = Color(10, 190, 70)
        val color3 = Color(210, 40, 40)
        val color4 = Color(210, 160, 60)
        sourceImage.setRGB(0, 0, color1.rgb)
        sourceImage.setRGB(0, 1, color2.rgb)
        sourceImage.setRGB(1, 0, color3.rgb)
        sourceImage.setRGB(1, 1, color4.rgb)

        val targetImage = ImageConverter.toNormalizedFloatArray(sourceImage, targetColorOrder)

        val expectedImage = FloatArray(sourceImage.getShape().numberOfElements.toInt()) { 0f }
        expectedImage.setRGB(0, 0, color1, sourceImage.getShape(), targetColorOrder)
        expectedImage.setRGB(0, 1, color2, sourceImage.getShape(), targetColorOrder)
        expectedImage.setRGB(1, 0, color3, sourceImage.getShape(), targetColorOrder)
        expectedImage.setRGB(1, 1, color4, sourceImage.getShape(), targetColorOrder)
        Assertions.assertArrayEquals(expectedImage, targetImage)
    }
}