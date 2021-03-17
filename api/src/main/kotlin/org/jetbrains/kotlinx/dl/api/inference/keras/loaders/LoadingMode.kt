/*
 * Copyright 2020 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.inference.keras.loaders

/** Loading strategy for remote loading. */
public enum class LoadingMode {
    SKIP_LOADING_IF_EXISTS,

    OVERRIDE_IF_EXISTS
}
