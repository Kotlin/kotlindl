/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.dataset.preprocessor

/** Helper class to keep widely used shape of image object presented as a 4D tensor (batchSize = 1, [width], [height], [channels]). */

// TODO: add the flag (channelsLast = true) + getter shape in format (1, HWC) or (1, CHW)
public class ImageShape(
    /** */
    public val width: Long? = null,
    /** */
    public val height: Long? = null,
    /** */
    public val channels: Long = 3
) {
    /** Returns number of elements in tensor with the given shape. */
    public val numberOfElements: Long
        get() = width!! * height!! * channels
}
