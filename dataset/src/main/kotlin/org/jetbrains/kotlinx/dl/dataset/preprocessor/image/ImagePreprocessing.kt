/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

import org.jetbrains.kotlinx.dl.dataset.image.ColorMode

/**
 * The whole image preprocessing pipeline DSL.
 *
 * It supports operations that implement [ImagePreprocessor], for example:
 * - [crop] See [Cropping] image preprocessor.
 * - [resize] See [Resize] image preprocessor.
 * - [rotate] See [Rotate] image preprocessor.
 *
 * It's a part of the [org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing] pipeline DSL.
 */
public class ImagePreprocessing {
    /** The internal state of [ImagePreprocessing]. */
    internal val operations = mutableListOf<ImagePreprocessor>()

    /** Adds a new operation to the [operations]. */
    public fun addOperation(operation: ImagePreprocessor) {
        operations.add(operation)
    }
}

/** */
public fun ImagePreprocessing.rotate(block: Rotate.() -> Unit) {
    addOperation(Rotate().apply(block))
}

/** */
public fun ImagePreprocessing.crop(block: Cropping.() -> Unit) {
    addOperation(Cropping().apply(block))
}

/** */
public fun ImagePreprocessing.resize(block: Resize.() -> Unit) {
    addOperation(Resize().apply(block))
}

/** */
public fun ImagePreprocessing.pad(block: Padding.() -> Unit) {
    addOperation(Padding().apply(block))
}

/** */
public fun ImagePreprocessing.convert(block: Convert.() -> Unit) {
    addOperation(Convert().apply(block))
}

/** */
public fun ImagePreprocessing.grayscale() {
    addOperation(Convert(colorMode = ColorMode.GRAYSCALE))
}

public fun ImagePreprocessing.centerCrop(block: CenterCrop.() -> Unit) {
    addOperation(CenterCrop().apply(block))
}