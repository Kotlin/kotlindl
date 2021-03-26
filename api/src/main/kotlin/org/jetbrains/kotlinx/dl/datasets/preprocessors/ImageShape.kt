/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.datasets.preprocessors

public class ImageShape(public val width: Long, public val height: Long, public val channels: Long = 3) {
    public val numberOfElements: Long
        get() = width * height * channels
}
