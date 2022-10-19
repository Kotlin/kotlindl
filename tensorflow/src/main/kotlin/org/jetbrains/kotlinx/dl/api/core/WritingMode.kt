/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core

/** Model writing mode. */
public enum class WritingMode {
    /** Throws an exception if directory exists. */
    FAIL_IF_EXISTS,

    /** Overrides directory if directory exists. */
    OVERRIDE,

    /** Append data to the directory if directory exists. */
    APPEND
}
