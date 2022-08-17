/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

import org.jetbrains.kotlinx.dl.api.core.shape.TensorShape
import org.jetbrains.kotlinx.dl.dataset.DataLoader
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape.Companion.toTensorShape
import java.io.File

/**
 * A [DataLoader] which uses provided [Preprocessing] to prepare images.
 */
private class PreprocessingDataLoader(private val preprocessing: Preprocessing) : DataLoader<File> {
    override fun load(dataSource: File): Pair<FloatArray, TensorShape> {
        val (floats, imageShape) = preprocessing(dataSource)
        return floats to imageShape.toTensorShape()
    }
}

/**
 * Returns a [DataLoader] instance which uses this [Preprocessing] to prepare images.
 */
public fun Preprocessing.dataLoader(): DataLoader<File> = PreprocessingDataLoader(this)