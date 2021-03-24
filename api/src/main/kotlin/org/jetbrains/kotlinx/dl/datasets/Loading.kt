/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

import org.jetbrains.kotlinx.dl.datasets.image.ColorOrder
import org.jetbrains.kotlinx.dl.datasets.image.ImageConverter
import java.io.File
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.streams.toList


public class Loading(
    public val dirLocation: String,
    public val imageShape: ImageShape,
    public val colorMode: ColorOrder
) : ImagePreprocessor {
    override fun apply(image: FloatArray, inputShape: ImageShape): Pair<FloatArray, ImageShape> {
        TODO("Not yet implemented")
    }

    internal fun fileToImage(file: File): FloatArray {
        return ImageConverter.toRawFloatArray(file, colorOrder = colorMode)
    }

    internal fun fileTo2D(file: File): Array<Array<FloatArray>> {
        val image = ImageConverter.getImage(file.inputStream())
        return ImageConverter.imageTo3DFloatArray(image)
    }

    public fun prepareFileNames(): Array<File> {
        return Files.list(Paths.get(dirLocation))
            .filter { path: Path -> Files.isRegularFile(path) }
            .filter { path: Path -> path.toString().endsWith(".jpg") || path.toString().endsWith(".png") }
            .map { obj: Path -> obj.toFile() }
            .toList()
            .toTypedArray()
    }
}
