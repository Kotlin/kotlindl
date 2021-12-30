package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImageFormatType

public interface ImageLoadingFacade{

	public var loadingStrategy: LoadingStrategy?

	public var imageShape: ImageShape?

	public var labelGenerator: LabelGenerator?

	public val imageFormat: ImageFormatType

	public val imageName: String

	public fun getImage(): BufferedImage

	public fun setImage(newImage: BufferedImage)

	public fun getImages(): Collection<BufferedImage>

	public fun getImageNames(): Collection<String>

	public fun save()
}