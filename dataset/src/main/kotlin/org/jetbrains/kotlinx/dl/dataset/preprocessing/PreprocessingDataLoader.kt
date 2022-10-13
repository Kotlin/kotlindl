/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessing

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.impl.preprocessing.image.ImageConverter
import java.awt.image.BufferedImage
import java.io.File
import java.io.InputStream

/**
 * A [DataLoader] which loads images from files and uses provided [Operation] to process them.
 */
private class PreprocessingFileDataLoader(
    private val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
) : DataLoader<File> {
    override fun load(dataSource: File): Pair<FloatArray, TensorShape> {
        require(dataSource.exists()) { "File '$dataSource' does not exist." }
        require(dataSource.isFile) {
            if (dataSource.isDirectory) "File '$dataSource' is a directory."
            else "File '$dataSource' is not a normal file."
        }
        return preprocessing.apply(ImageConverter.toBufferedImage(dataSource))
    }
}

/**
 * A [DataLoader] which loads images from input streams and uses provided [Operation] to process them.
 */
private class PreprocessingInputStreamDataLoader(
    private val preprocessing: Operation<BufferedImage, Pair<FloatArray, TensorShape>>
) : DataLoader<InputStream> {
    override fun load(dataSource: InputStream): Pair<FloatArray, TensorShape> {
        return preprocessing.apply(ImageConverter.toBufferedImage(dataSource))
    }
}

/**
 * Returns a [DataLoader] instance which loads images from files and uses this [Operation] to process them.
 */
public fun Operation<BufferedImage, Pair<FloatArray, TensorShape>>.fileLoader(): DataLoader<File> =
    PreprocessingFileDataLoader(this)

/**
 * Returns a [DataLoader] instance which loads images from input streams and uses this [Operation] to process them.
 */
public fun Operation<BufferedImage, Pair<FloatArray, TensorShape>>.inputStreamLoader(): DataLoader<InputStream> =
    PreprocessingInputStreamDataLoader(this)