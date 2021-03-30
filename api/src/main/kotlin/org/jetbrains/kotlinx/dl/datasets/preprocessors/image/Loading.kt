/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.preprocessors.image

import org.jetbrains.kotlinx.dl.datasets.image.ColorOrder
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import org.jetbrains.kotlinx.dl.datasets.preprocessors.ImageShape
import java.awt.image.BufferedImage
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import kotlin.streams.toList


public class Loading(
    public var pathToData: File? = null,
    public var imageShape: ImageShape = ImageShape(32, 32, 3),
    /** Keep channels in the given order after loading. */
    public var colorMode: ColorOrder = ColorOrder.BGR
) : ImagePreprocessor {


    internal fun fileToImage(file: File): BufferedImage {
        return ImageConverter.toBufferedImage(file.inputStream(), colorOrder = colorMode)
    }

    /*internal fun fileTo2D(file: File): Array<Array<FloatArray>> {
        val image = ImageConverter.getImage(file.inputStream())
        return ImageConverter.imageTo3DFloatArray(image)
    }*/

    public fun prepareFileNames(): Array<File> {
        return Files.list(pathToData!!.toPath())
            .filter { path: Path -> Files.isRegularFile(path) }
            .filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
            .map { obj: Path -> obj.toFile() }
            .toList()
            .toTypedArray()
    }

    override fun apply(image: BufferedImage, inputShape: ImageShape): Pair<BufferedImage, ImageShape> {
        TODO("Not yet implemented")
    }
}
