/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.getShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImagePreprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading.ImageLoadingFacade
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading.ImageLoadingStrategy
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading.ImageLoadingFacadeImpl

/**
 * The data preprocessing pipeline presented as Kotlin DSL on receivers.
 *
 * Could be used to handle directory of images or one image file.
 */
public class Preprocessing {
	/** */
	public var load: ImageLoadingFacade? = null

	/** This stage describes the process of image loading and transformation before converting to tensor. */
	public var imagePreprocessingStage: ImagePreprocessing? = null

	/** This stage describes the process of data transformation after converting to tensor. */
	public var tensorPreprocessingStage: TensorPreprocessing? = null

	/** Returns the final shape of data when image preprocessing is applied to the image. */
	public val finalShape: ImageShape
		get() {
			var imageShape = load?.imageShape

			imagePreprocessingStage?.operations?.forEach { operation ->
				imageShape = operation.getOutputShape(imageShape)
			}

			return getImageShapeOrThrow(imageShape)
		}

	private fun getImageShapeOrThrow(imageShape: ImageShape?): ImageShape =
		imageShape ?: throw IllegalStateException(
			"Final image shape is unclear. Operator with fixed output size (such as \"resize\") should be used " +
					"or imageShape with height, weight and channels should be initialized."
		)

	/** Applies the preprocessing pipeline to the specific image file. */
	public operator fun invoke(): Pair<FloatArray, ImageShape> {
		val immutableImageLoading = load
		requireNotNull(immutableImageLoading)

		requireNotNull(immutableImageLoading.loadingStrategy as? ImageLoadingStrategy) { "Invoke call is available for single image preprocessing only." }

		return preprocessImage(immutableImageLoading)
	}

	internal fun preprocessImage(imageLoadingFacade: ImageLoadingFacade): Pair<FloatArray, ImageShape> {
		imagePreprocessingStage?.operations?.forEach { operation ->
			imageLoadingFacade.setImage(newImage = operation.apply(imageLoadingFacade.getImage()))
			imageLoadingFacade.save()
		}

		var tensor = ImageConverter.toRawFloatArray(imageLoadingFacade.getImage())
		val shape = imageLoadingFacade.getImage().getShape()

		tensorPreprocessingStage?.operations?.forEach { operation ->
			tensor = operation.apply(tensor, shape)
		}

		return tensor to shape
	}
}

/** */
public fun preprocess(init: Preprocessing.() -> Unit): Preprocessing =
	Preprocessing()
		.apply(init)

/** */
public fun Preprocessing.load(block: ImageLoadingFacade.() -> Unit) {
	load = ImageLoadingFacadeImpl().apply(block)
}

/** */
public fun Preprocessing.transformImage(block: ImagePreprocessing.() -> Unit) {
	imagePreprocessingStage = ImagePreprocessing().apply(block)
}

/** */
public fun Preprocessing.transformTensor(block: TensorPreprocessing.() -> Unit) {
	tensorPreprocessingStage = TensorPreprocessing().apply(block)
}