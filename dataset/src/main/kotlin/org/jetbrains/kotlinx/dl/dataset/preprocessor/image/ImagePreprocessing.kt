/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor.image

/**
 * The whole image preprocessing pipeline DSL.
 *
 * It supports the following ops:
 * - [crop] See [Cropping] image preprocessor.
 * - [resize] See [Resize] image preprocessor.
 * - [rotate] See [Rotate] image preprocessor.
 * - [save] See [Save] image preprocessor.
 *
 * It's a part of the [org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing] pipeline DSL.
 */
public class ImagePreprocessing {
    /** */
    public lateinit var load: Loading

    /** */
    public lateinit var crop: Cropping

    /** */
    public lateinit var resize: Resize

    /** */
    public lateinit var rotate: Rotate

    /** */
    public lateinit var save: Save

    /** True, if [crop] is initialized. */
    public val isCropInitialized: Boolean
        get() = ::crop.isInitialized

    /** True, if [resize] is initialized. */
    public val isResizeInitialized: Boolean
        get() = ::resize.isInitialized

    /** True, if [rotate] is initialized. */
    public val isRotateInitialized: Boolean
        get() = ::rotate.isInitialized

    /** True, if [load] is initialized. */
    public val isLoadInitialized: Boolean
        get() = ::load.isInitialized

    /** True, if [save] is initialized. */
    public val isSaveInitialized: Boolean
        get() = ::save.isInitialized
}

/** */
public fun ImagePreprocessing.load(block: Loading.() -> Unit) {
    load = Loading().apply(block)
}

/** */
public fun ImagePreprocessing.rotate(block: Rotate.() -> Unit) {
    rotate = Rotate().apply(block)
}

/** */
public fun ImagePreprocessing.crop(block: Cropping.() -> Unit) {
    crop = Cropping().apply(block)
}

/** */
public fun ImagePreprocessing.resize(block: Resize.() -> Unit) {
    resize = Resize().apply(block)
}

/** */
public fun ImagePreprocessing.save(block: Save.() -> Unit) {
    save = Save().apply(block)
}




