package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Normalizing
import org.jetbrains.kotlinx.dl.dataset.preprocessing.mean
import org.jetbrains.kotlinx.dl.dataset.preprocessing.std
import org.junit.jupiter.api.Assertions
import org.junit.jupiter.api.Test
import java.awt.Color
import java.awt.image.BufferedImage

class PreprocessingTensorTest {
    @Test
    fun normalizeTest() {
        val inputImage = BufferedImage(2, 2, BufferedImage.TYPE_3BYTE_BGR)
        inputImage.setRGB(0, 0, Color(50, 150, 200).rgb)
        inputImage.setRGB(0, 1, Color(10, 190, 70).rgb)
        inputImage.setRGB(1, 0, Color(210, 40, 40).rgb)
        inputImage.setRGB(1, 1, Color(210, 160, 60).rgb)

        val imageFloats = ImageConverter.toNormalizedFloatArray(inputImage)

        val (normalizedImage, _) = Normalizing().apply {
            mean = imageFloats.mean(channels = 3)
            std = imageFloats.std(channels = 3)
        }.apply(imageFloats to TensorShape(2, 2, 3))

        Assertions.assertArrayEquals(FloatArray(3) { 0f }, normalizedImage.mean(3), EPS)
        Assertions.assertArrayEquals(FloatArray(3) { 1f }, normalizedImage.std(3), EPS)
    }

    companion object {
        private const val EPS: Float = 2e-7f
    }
}