/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer.convolutional

/**
 * Type of padding.
 */
public enum class ConvPadding {
    /**
     * Results in padding evenly to the left/right or up/down of the input such that output has the same
     * height/width dimension as the input.
     */
    SAME,

    /** No padding. */
    VALID,

    /** Full padding. For Keras compatibility goals. */
    FULL
}
