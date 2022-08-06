/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode

/**
 * Represents the image preprocessing stage of the [org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing].
 * Consists of the operations implementing [ImagePreprocessing] which are applied to the image one by one.
 *
 * Supported operations include:
 * - [crop] and [centerCrop],
 * - [resize],
 * - [rotate],
 * - [convert] and [grayscale],
 * - [pad].
 *
 * @see Cropping
 * @see CenterCrop
 * @see Resize
 * @see Rotate
 * @see Convert
 * @see Padding
 */
public class ImagePreprocessing {
    /** The internal state of [ImagePreprocessing]. */
    internal val operations = mutableListOf<ImagePreprocessor>()

    /** Adds a new operation to the [operations]. */
    public fun addOperation(operation: ImagePreprocessor) {
        operations.add(operation)
    }
}

/** Applies [Rotate] operation to rotate the image by an arbitrary angle (specified in degrees). */
public fun ImagePreprocessing.rotate(block: Rotate.() -> Unit) {
    addOperation(Rotate().apply(block))
}

/** Applies [Cropping] operation to crop the image by the specified amount. */
public fun ImagePreprocessing.crop(block: Cropping.() -> Unit) {
    addOperation(Cropping().apply(block))
}

/** Applies [Resize] operation to resize the image to a specific size. */
public fun ImagePreprocessing.resize(block: Resize.() -> Unit) {
    addOperation(Resize().apply(block))
}

/** Applies [Padding] operation to pad the image. */
public fun ImagePreprocessing.pad(block: Padding.() -> Unit) {
    addOperation(Padding().apply(block))
}

/** Applies [Convert] operation to convert the image to a different [ColorMode]. */
public fun ImagePreprocessing.convert(block: Convert.() -> Unit) {
    addOperation(Convert().apply(block))
}

/** Applies [Convert] operation to convert the image to [ColorMode.GRAYSCALE]. */
public fun ImagePreprocessing.grayscale() {
    addOperation(Convert(colorMode = ColorMode.GRAYSCALE))
}

/** Applies [CenterCrop] operation to crop the image at the center. */
public fun ImagePreprocessing.centerCrop(block: CenterCrop.() -> Unit) {
    addOperation(CenterCrop().apply(block))
}