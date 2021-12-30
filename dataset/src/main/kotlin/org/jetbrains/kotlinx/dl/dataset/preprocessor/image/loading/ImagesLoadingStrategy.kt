package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage

public interface ImagesLoadingStrategy : LoadingStrategy {

	public fun getImages(): Collection<BufferedImage>

	public fun getImageNames(): Collection<String>
}