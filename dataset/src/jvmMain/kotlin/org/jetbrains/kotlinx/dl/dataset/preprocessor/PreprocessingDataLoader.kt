/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.dataset.image.ImageConverter
import org.jetbrains.kotlinx.dl.dataset.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape.Companion.toTensorShape
import java.awt.image.BufferedImage
import java.io.File
import java.nio.channels.FileLock

/**
 * A [DataLoader] which uses provided [Preprocessing] to prepare images.
 */
private class PreprocessingDataLoader(
    private val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
) : DataLoader<File> {
    override fun load(dataSource: File): Pair<FloatArray, TensorShape> {
        require(dataSource.exists()) { "File '$dataSource' does not exist." }
        require(dataSource.isFile) {
            if (dataSource.isDirectory) "File '$dataSource' is a directory."
            else "File '$dataSource' is not a normal file."
        }

        val image = dataSource.inputStream().use { inputStream -> ImageConverter.toBufferedImage(inputStream) }

        val (floats, imageShape) = preprocessing.apply(image)
        return floats to imageShape
    }
}

/**
 * Returns a [DataLoader] instance which uses this [Preprocessing] to prepare images.
 */
public fun Operation<BufferedImage, Pair<FloatArray, TensorShape>>.dataLoader(): DataLoader<File> =
    PreprocessingDataLoader(this)
