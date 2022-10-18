/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.impl.preprocessing.image

/**
 * The speed of single file preprocessing could be tuned via this setting.
 */
public enum class RenderingSpeed {
    /** */
    FAST,

    /** */
    SLOW,

    /** */
    MEDIUM
}
