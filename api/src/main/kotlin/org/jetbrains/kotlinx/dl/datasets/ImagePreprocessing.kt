/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets

public class ImagePreprocessing {
    public lateinit var load: Loading
    public lateinit var crop: Cropping
    public lateinit var resize: Resize
    public lateinit var rotate: Rotate
}

public fun imagePreprocessing(init: ImagePreprocessing.() -> Unit): ImagePreprocessing =
    ImagePreprocessing()
        .apply(init)

public fun ImagePreprocessing.load(block: Loading.() -> Unit) {
    load = Loading().apply(block)
}

public fun ImagePreprocessing.rotate(block: Rotate.() -> Unit) {
    rotate = Rotate().apply(block)
}

public fun ImagePreprocessing.crop(block: Cropping.() -> Unit) {
    crop = Cropping().apply(block)
}

public fun ImagePreprocessing.resize(block: Resize.() -> Unit) {
    resize = Resize().apply(block)
}




