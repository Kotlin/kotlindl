/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

/**
 * Represents the number and order of color channels in the image.
 *
 * @property [channels] number of image channels
 * */
public enum class ColorMode(public val channels: Int) {
    /** Red, green, blue. */
    RGB(3),

    /** Blue, green, red. */
    BGR(3),

    /** Grayscale **/
    GRAYSCALE(1)
}