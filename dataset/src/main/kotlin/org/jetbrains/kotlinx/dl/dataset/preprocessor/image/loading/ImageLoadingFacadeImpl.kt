package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImageFormatType

// TODO : Find a cleaner way to abstract image and images entrypoint (try to get rid of !!)
public class ImageLoadingFacadeImpl public constructor(
	override var loadingStrategy: LoadingStrategy? = null,
	override var imageShape: ImageShape? = null,
	override var labelGenerator: LabelGenerator? = null,
) : ImageLoadingFacade {

	override val imageFormat: ImageFormatType
		get() = (loadingStrategy as? ImageLoadingStrategy)!!.imageFormat

	override val imageName: String
		get() = (loadingStrategy as? ImageLoadingStrategy)!!.imageName

	override fun getImage(): BufferedImage {
		return (loadingStrategy as? ImageLoadingStrategy)!!.getImage()
	}

	override fun setImage(newImage: BufferedImage) {
		(loadingStrategy as? ImageLoadingStrategy)!!.setImage(newImage)
	}

	override fun getImages(): Collection<BufferedImage> {
		return (loadingStrategy as? ImagesLoadingStrategy)!!.getImages()
	}

	override fun getImageNames(): Collection<String> {
		return (loadingStrategy as? ImagesLoadingStrategy)!!.getImageNames()
	}

	override fun save() {
		loadingStrategy?.save()
	}

}