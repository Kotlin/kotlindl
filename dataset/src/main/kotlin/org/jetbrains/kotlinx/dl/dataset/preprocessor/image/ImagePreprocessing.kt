/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

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
    /** */
    internal val operations = mutableListOf<ImagePreprocessor>()

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