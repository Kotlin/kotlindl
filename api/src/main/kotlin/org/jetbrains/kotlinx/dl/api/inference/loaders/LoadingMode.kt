/*
 * Copyright 2020-2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.loaders

/** Loading strategy for remote loading. */
public enum class LoadingMode {
    /** Skip loading if exists. */
    SKIP_LOADING_IF_EXISTS,

    /** Overrides if exists. */
    OVERRIDE_IF_EXISTS
}
