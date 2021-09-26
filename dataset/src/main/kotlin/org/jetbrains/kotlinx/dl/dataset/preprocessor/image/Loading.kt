/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorOrder
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.generator.LabelGenerator
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.toList

/**
 * This class defines the Loading operation.
 *
 * Describes the initial phase of data and its labels loading.
 * Could be applied to one file or whole directory.
 *
 * @property [pathToData] The image will be cropped from the top by the given number of pixels.
 * @property [imageShape] The shape of input image. If height and width are different for different images, need to skip the filling of these properties of [ImageShape].
 * @property [labelGenerator] A way to generate labels.
 * @property [colorMode] Color mode.
 *
 * NOTE: currently it supports [BufferedImage.TYPE_3BYTE_BGR] image type only.
 */
public class Loading(
    public var pathToData: File? = null,
    public var imageShape: ImageShape? = null,
    public var labelGenerator: LabelGenerator? = null,
    /** Keep channels in the given order after loading. */
    public var colorMode: ColorOrder = ColorOrder.BGR
) {
    internal fun fileToImage(file: File): BufferedImage {
        return file.inputStream().use { inputStream -> ImageConverter.toBufferedImage(inputStream) }
    }

    /*internal fun fileTo2D(file: File): Array<Array<FloatArray>> {
        val image = ImageConverter.getImage(file.inputStream())
        return ImageConverter.imageTo3DFloatArray(image)
    }*/

    /** Returns array of files found in the [pathToData] directory. */
    internal fun prepareFileNames(): Array<File> {
        return Files.walk(pathToData!!.toPath())
            .filter { path: Path -> Files.isRegularFile(path) }
            .filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
            .map { obj: Path -> obj.toFile() }
            .toList()
            .toTypedArray()
    }
}
