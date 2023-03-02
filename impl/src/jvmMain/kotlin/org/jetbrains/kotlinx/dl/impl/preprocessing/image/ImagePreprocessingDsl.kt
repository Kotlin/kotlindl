/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

import org.jetbrains.kotlinx.dl.api.core.FloatData
import org.jetbrains.kotlinx.dl.api.preprocessing.Operation
import org.jetbrains.kotlinx.dl.api.preprocessing.PreprocessingPipeline
import java.awt.image.BufferedImage

/**
 * Extension function for building preprocessing pipeline for images.
 */

/** Applies [Rotate] operation to rotate the image by an arbitrary angle (specified in degrees). */
public fun <I> Operation<I, BufferedImage>.rotate(block: Rotate.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Rotate().apply(block))
}

/** Applies [Cropping] operation to crop the image by the specified amount. */
public fun <I> Operation<I, BufferedImage>.crop(block: Cropping.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Cropping().apply(block))
}

/** Applies [Resize] operation to resize the image to a specific size. */
public fun <I> Operation<I, BufferedImage>.resize(block: Resize.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Resize().apply(block))
}

/** Applies [Padding] operation to pad the image. */
public fun <I> Operation<I, BufferedImage>.pad(block: Padding.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Padding().apply(block))
}

/** Applies [Convert] operation to convert the image to a different [ColorMode]. */
public fun <I> Operation<I, BufferedImage>.convert(block: Convert.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Convert().apply(block))
}

/** Applies [Convert] operation to convert the image to [ColorMode.GRAYSCALE]. */
public fun <I> Operation<I, BufferedImage>.grayscale(): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, Convert(colorMode = ColorMode.GRAYSCALE))
}

/** Applies [CenterCrop] operation to crop the image at the center. */
public fun <I> Operation<I, BufferedImage>.centerCrop(block: CenterCrop.() -> Unit): Operation<I, BufferedImage> {
    return PreprocessingPipeline(this, CenterCrop().apply(block))
}

/** Applies [ConvertToFloatArray] operation to convert the image to a float array. */
public fun <I> Operation<I, BufferedImage>.toFloatArray(block: ConvertToFloatArray.() -> Unit): Operation<I, FloatData> {
    return PreprocessingPipeline(this, ConvertToFloatArray().apply(block))
}
