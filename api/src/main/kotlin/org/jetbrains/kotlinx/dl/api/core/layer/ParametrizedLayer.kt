/*
 * Copyright 2022 JetBrains s.r.o. and Kotlin Deep Learning project contributors. All Rights Reserved.
 * Use of this source code is governed by the Apache 2.0 license that can be found in the LICENSE.txt file.
 */

package org.jetbrains.kotlinx.dl.api.core.layer

/**
 * Represents a [Layer] with parameters.
 */
public interface ParametrizedLayer {
    /** Number of parameters in this layer. */
    public val paramCount: Int
}

/**
 * Returns the number of parameters in this layer. If layer is not a [ParametrizedLayer], returns zero.
 */
public val Layer.paramCount: Int
    get() = if (this is ParametrizedLayer) paramCount else 0