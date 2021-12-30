/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import java.io.File
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.image.toByteArray
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.ImageFormatType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.Saver


public class FileLoadingStrategy(
	override val imageFormat: ImageFormatType,
	public var imageFile: File,
) : ImageLoadingStrategy {

	init {
		require(imageFile.isFile) { "Expected file, got folder instead!" }
	}

	override val imageName: String
		get() = imageFile.name

	override fun getImage(): BufferedImage = ImageConverter.toBufferedImage(imageFile.inputStream())

	override fun setImage(newImage: BufferedImage) {
		imageFile.writeBytes(newImage.toByteArray(imageFormat))
	}

	override fun save() {
		Saver().save(imageName = imageFile.name, image = getImage())
	}
}
