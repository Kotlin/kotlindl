/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image.loading

import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.toList
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.Saver

public class DirectoryLoadingStrategy(
	public var imagesFolder: File,
) : ImagesLoadingStrategy {

	init {
		require(imagesFolder.isDirectory) { "Expected directory, got file instead!" }
	}

	override fun getImages(): Collection<BufferedImage> =
		getImagesFiles().map(::getImage)

	private fun getImagesFiles(): Collection<File> =
		Files.walk(imagesFolder.toPath())
			.filter { path: Path -> Files.isRegularFile(path) }
			.filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
			.map { it.toFile() }
			.toList()

	private fun getImage(file: File) = ImageConverter.toBufferedImage(file.inputStream())

	override fun getImageNames(): Collection<String> = getImagesFiles().map { it.name }

	override fun save() {

		val saver = Saver()

		getImagesFiles().forEach {
			saver.save(imageName = it.name, image = getImage(it))
		}
	}
}
