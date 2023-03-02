/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

/**
 * When an image is resized, we need an interpolation algorithm to draw newly created pixels between old pixels.
 *
 * Conceptually the image is viewed as a set of infinitely small
 * point color samples which have value only at the centers of
 * integer coordinate pixels and the space between those pixel
 * centers is filled with linear ramps of colors that connect
 * adjacent discrete samples in a straight line.
 */
public enum class InterpolationType {
    /**
     * The color samples of the 4 nearest
     * neighboring integer coordinate samples in the image are
     * interpolated linearly to produce a color sample.
     *
     * @see [java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR]
     */
    BILINEAR,

    /**
     * The color samples of 9 nearby
     * integer coordinate samples in the image are interpolated using
     * a cubic function in both ```X``` and ```Y``` to produce
     * a color sample.
     *
     * @see [java.awt.RenderingHints.VALUE_INTERPOLATION_BICUBIC]
     */
    BICUBIC,

    /**
     * The color sample of the nearest
     * neighboring integer coordinate sample in the image is used.
     *
     * @see [java.awt.RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR]
     */
    NEAREST
}
