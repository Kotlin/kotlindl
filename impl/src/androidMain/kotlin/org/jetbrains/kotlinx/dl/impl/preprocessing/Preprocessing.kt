/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing

import android.graphics.Bitmap
import android.os.Build
import androidx.annotation.RequiresApi
import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline
import org.jetbrains.kotlinx.dl.impl.preprocessing.bitmap.Crop
import org.jetbrains.kotlinx.dl.impl.preprocessing.bitmap.Resize
import org.jetbrains.kotlinx.dl.impl.preprocessing.bitmap.Rotate

/**
 * The data preprocessing pipeline presented as Kotlin DSL on receivers.
 */

/** Applies [ConvertToFloatArray] operation to convert the [Bitmap] to a float array. */
public fun <I> Operation<I, Bitmap>.toFloatArray(block: ConvertToFloatArray.() -> Unit): Operation<I, FloatData> {
    return PreprocessingPipeline(this, ConvertToFloatArray().apply(block))
}

/** Applies [Resize] operation to resize the [Bitmap] to a specific size. */
public fun <I> Operation<I, Bitmap>.resize(block: Resize.() -> Unit): Operation<I, Bitmap> {
    return PreprocessingPipeline(this, Resize().apply(block))
}

/** Applies [Rotate] operation to rotate the [Bitmap] by an arbitrary angle (specified in degrees). */
public fun <I> Operation<I, Bitmap>.rotate(block: Rotate.() -> Unit): Operation<I, Bitmap> {
    return PreprocessingPipeline(this, Rotate().apply(block))
}

/** Applies [Crop] operation to crop the [Bitmap] at a specified region. */
@RequiresApi(Build.VERSION_CODES.O)
public fun <I> Operation<I, Bitmap>.crop(block: Crop.() -> Unit): Operation<I, Bitmap> {
    return PreprocessingPipeline(this, Crop().apply(block))
}