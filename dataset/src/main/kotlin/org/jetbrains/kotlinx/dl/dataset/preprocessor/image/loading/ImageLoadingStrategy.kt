package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImageFormatType

public interface ImageLoadingStrategy: LoadingStrategy {

	public val imageFormat: ImageFormatType

	public val imageName: String

	public fun getImage(): BufferedImage

	public fun setImage(newImage: BufferedImage)
}