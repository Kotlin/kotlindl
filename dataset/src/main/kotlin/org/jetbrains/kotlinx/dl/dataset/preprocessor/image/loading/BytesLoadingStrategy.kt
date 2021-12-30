package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import java.io.ByteArrayInputStream
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.toByteArray
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImageFormatType


public class BytesLoadingStrategy(
	override val imageFormat: ImageFormatType,
	private var imageBytes: ByteArray,
	imageNameValue: String? = null,
) : ImageLoadingStrategy {

	override val imageName: String = imageNameValue ?: hashCode().toString()

	override fun getImage(): BufferedImage =
		ImageConverter.toBufferedImage(ByteArrayInputStream(imageBytes))

	override fun setImage(newImage: BufferedImage) {
		imageBytes = newImage.toByteArray(imageFormat)
	}

	override fun save() {}
}